"""
vulkan_query.py — Vulkan compute replacement for broken OpenCL query kernel.

Why Vulkan and not OpenCL:
    AMD Adrenalin's OpenCL driver (PAL,LC / gfx1201) uses the PAL-LLVM
    compiler backend which has a documented optimizer bug that corrupts
    variable-depth loop SHA256 kernels (wrong fingerprints, false hits).
    Vulkan's compute pipeline uses the ACO compiler backend, which is a
    completely separate code path and does not have this bug.

Install:
    pip install vulkan
    Vulkan SDK in PATH (winget install KhronosGroup.VulkanSDK)

Usage:
    from vulkan_query import VulkanQueryEngine, vk_available
    if vk_available():
        eng = VulkanQueryEngine(chain_len, charset, str_len)
        W, wv, mm = eng.query_batch(target_bytes, batch_offset, n)
        eng.destroy()
"""

import os
import sys
import subprocess
import tempfile
import ctypes
import numpy as np

# ── Availability check ────────────────────────────────────────────────────

def vk_available() -> bool:
    try:
        import vulkan as vk  # noqa
        return True
    except ImportError:
        return False


# ── GLSL Compute Shader ──────────────────────────────────────────────────
# Implements identical logic to py_reduce_inj + sha256h in Python.
# Uses GL_ARB_gpu_shader_int64 for 64-bit seed arithmetic (supported on
# all RDNA-class AMD hardware). Variable loop start (step=pos) is safe
# here because ACO does not have the PAL-LLVM optimizer bug.

# Shader loaded from cdp_query.comp (same directory as this file)
_GLSL_SHADER = open(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cdp_query.comp'),
    encoding='utf-8'
).read()

def _compile_glsl_to_spirv(glsl: str) -> bytes:
    """Compile GLSL compute shader to SPIR-V using glslangValidator."""
    with tempfile.NamedTemporaryFile(suffix='.comp', mode='w',
                                     delete=False, encoding='utf-8') as f:
        f.write(glsl)
        glsl_path = f.name
    spv_path = glsl_path + '.spv'
    try:
        result = subprocess.run(
            ['glslangValidator', '-V', '--target-env', 'vulkan1.0',
             glsl_path, '-o', spv_path],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"GLSL compilation failed:\n{result.stdout}\n{result.stderr}"
            )
        with open(spv_path, 'rb') as f:
            return f.read()
    finally:
        try: os.unlink(glsl_path)
        except: pass
        try: os.unlink(spv_path)
        except: pass


class VulkanQueryEngine:
    """
    Vulkan compute replacement for OpenCL query kernel.

    Sets up Vulkan instance → device → pipeline once at init.
    query_batch() dispatches one batch and returns fingerprints.
    Call destroy() when done (or use as context manager).
    """

    BATCH_MAX = 65536  # maximum threads per dispatch

    def __init__(self, chain_len: int, charset: str, str_len: int,
                 device_index: int = 0, verbose: bool = True):
        import vulkan as vk
        self._vk = vk
        self.chain_len   = chain_len
        self.charset     = charset
        self.str_len     = str_len
        self._verbose    = verbose
        self._destroyed  = False

        if verbose:
            print("  [Vulkan] Compiling compute shader (ACO backend)...")
        self._spirv = _compile_glsl_to_spirv(_GLSL_SHADER)

        self._setup(device_index)
        self._build_pipeline()
        self._alloc_buffers()
        self._build_descriptor_set()
        self._alloc_command_buffer()

        if verbose:
            props = vk.vkGetPhysicalDeviceProperties(self._pdev)
            dn   = props.deviceName
            name = dn.rstrip('\x00') if isinstance(dn, str) else bytes(dn).rstrip(b'\x00').decode()
            print(f"  [Vulkan] Ready on '{name}'  (Vulkan compute / ACO)")

    # ── Vulkan setup ──────────────────────────────────────────────────

    def _setup(self, preferred_index: int):
        vk = self._vk

        # Instance
        app_info = vk.VkApplicationInfo(
            pApplicationName   = b"cdp_vulkan_query",
            applicationVersion = vk.VK_MAKE_VERSION(1, 0, 0),
            pEngineName        = b"none",
            engineVersion      = vk.VK_MAKE_VERSION(1, 0, 0),
            apiVersion         = vk.VK_API_VERSION_1_0,
        )
        inst_info = vk.VkInstanceCreateInfo(pApplicationInfo=app_info)
        self._instance = vk.vkCreateInstance(inst_info, None)

        # Physical device — prefer AMD (vendorID 0x1002)
        pdevs = vk.vkEnumeratePhysicalDevices(self._instance)
        if not pdevs:
            raise RuntimeError("[Vulkan] No physical devices found")

        # Sort: AMD first, then by preferred index
        def amd_first(pd):
            props = vk.vkGetPhysicalDeviceProperties(pd)
            return (0 if props.vendorID == 0x1002 else 1)

        pdevs_sorted = sorted(pdevs, key=amd_first)
        self._pdev = pdevs_sorted[min(preferred_index, len(pdevs_sorted)-1)]

        # Find compute queue family
        qprops = vk.vkGetPhysicalDeviceQueueFamilyProperties(self._pdev)
        self._qfam = None
        for i, qp in enumerate(qprops):
            if qp.queueFlags & vk.VK_QUEUE_COMPUTE_BIT:
                self._qfam = i
                break
        if self._qfam is None:
            raise RuntimeError("[Vulkan] No compute queue family found")

        # Logical device
        queue_info = vk.VkDeviceQueueCreateInfo(
            queueFamilyIndex = self._qfam,
            queueCount       = 1,
            pQueuePriorities = [1.0],
        )
        dev_info = vk.VkDeviceCreateInfo(
            pQueueCreateInfos       = [queue_info],
            queueCreateInfoCount    = 1,
            enabledExtensionCount   = 0,
        )
        self._dev   = vk.vkCreateDevice(self._pdev, dev_info, None)
        self._queue = vk.vkGetDeviceQueue(self._dev, self._qfam, 0)

        # Memory type indices
        mem_props = vk.vkGetPhysicalDeviceMemoryProperties(self._pdev)
        self._mem_host_idx = self._find_memory_type(
            mem_props,
            vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        )

    def _find_memory_type(self, mem_props, flags) -> int:
        vk = self._vk
        for i in range(mem_props.memoryTypeCount):
            if (mem_props.memoryTypes[i].propertyFlags & flags) == flags:
                return i
        raise RuntimeError(f"[Vulkan] No suitable memory type for flags={flags:#x}")

    # ── Pipeline ──────────────────────────────────────────────────────

    def _build_pipeline(self):
        vk = self._vk

        # Shader module
        # vulkan._vulkan.py:6272 does ffi.cast('uint32_t*', ffi.from_buffer(v))
        # So pCode must be plain bytes — the package calls ffi.from_buffer itself.
        # bytes is truthy (passes 6261 filter) and buffer-compatible (passes 6272).
        mod_info = vk.VkShaderModuleCreateInfo(
            codeSize = len(self._spirv),
            pCode    = bytes(self._spirv),
        )
        self._shader_mod = vk.vkCreateShaderModule(self._dev, mod_info, None)

        # Descriptor set layout — 5 storage buffers
        bindings = [
            vk.VkDescriptorSetLayoutBinding(
                binding         = i,
                descriptorType  = vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount = 1,
                stageFlags      = vk.VK_SHADER_STAGE_COMPUTE_BIT,
            )
            for i in range(5)
        ]
        dsl_info = vk.VkDescriptorSetLayoutCreateInfo(
            bindingCount = len(bindings),
            pBindings    = bindings,
        )
        self._dsl = vk.vkCreateDescriptorSetLayout(self._dev, dsl_info, None)

        # Push constant range (4 ints = 16 bytes)
        pc_range = vk.VkPushConstantRange(
            stageFlags = vk.VK_SHADER_STAGE_COMPUTE_BIT,
            offset     = 0,
            size       = 16,  # 4 × int32
        )

        # Pipeline layout
        pl_info = vk.VkPipelineLayoutCreateInfo(
            setLayoutCount         = 1,
            pSetLayouts            = [self._dsl],
            pushConstantRangeCount = 1,
            pPushConstantRanges    = [pc_range],
        )
        self._pl = vk.vkCreatePipelineLayout(self._dev, pl_info, None)

        # Compute pipeline
        stage = vk.VkPipelineShaderStageCreateInfo(
            stage  = vk.VK_SHADER_STAGE_COMPUTE_BIT,
            module = self._shader_mod,
            pName  = b"main",
        )
        cp_info = vk.VkComputePipelineCreateInfo(
            stage  = stage,
            layout = self._pl,
        )
        self._pipeline = vk.vkCreateComputePipelines(
            self._dev, None, 1, [cp_info], None
        )[0]

    # ── Buffer allocation ─────────────────────────────────────────────

    def _make_buffer(self, size_bytes: int):
        """Create a host-visible/coherent buffer. Returns (VkBuffer, VkDeviceMemory)."""
        vk = self._vk
        buf_info = vk.VkBufferCreateInfo(
            size        = size_bytes,
            usage       = vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            sharingMode = vk.VK_SHARING_MODE_EXCLUSIVE,
        )
        buf = vk.vkCreateBuffer(self._dev, buf_info, None)
        reqs = vk.vkGetBufferMemoryRequirements(self._dev, buf)
        alloc_info = vk.VkMemoryAllocateInfo(
            allocationSize  = reqs.size,
            memoryTypeIndex = self._mem_host_idx,
        )
        mem = vk.vkAllocateMemory(self._dev, alloc_info, None)
        vk.vkBindBufferMemory(self._dev, buf, mem, 0)
        return buf, mem

    def _alloc_buffers(self):
        N = self.BATCH_MAX
        cs_bytes = self.charset.encode()

        # target_hash: 8 × uint32 = 32 bytes
        self._tgt_buf, self._tgt_mem = self._make_buffer(32)

        # charset: one uint32 per char
        self._cs_buf, self._cs_mem = self._make_buffer(len(cs_bytes) * 4)

        # outputs
        self._W_buf,  self._W_mem  = self._make_buffer(N * 4)
        self._wv_buf, self._wv_mem = self._make_buffer(N * 16 * 4)
        self._mm_buf, self._mm_mem = self._make_buffer(N * 2  * 4)

        # Write charset (stays constant)
        self._write_charset()

    def _write_charset(self):
        vk = self._vk
        cs_bytes = self.charset.encode()
        n = len(cs_bytes)
        data = struct.pack(f'{n}I', *cs_bytes)
        buf = vk.vkMapMemory(self._dev, self._cs_mem, 0, n * 4, 0)
        # cffi buffer supports slice assignment for writing
        buf[0:n * 4] = data
        vk.vkUnmapMemory(self._dev, self._cs_mem)

    # ── Descriptor set ────────────────────────────────────────────────

    def _build_descriptor_set(self):
        vk = self._vk

        # Pool
        pool_size = vk.VkDescriptorPoolSize(
            type            = vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            descriptorCount = 5,
        )
        pool_info = vk.VkDescriptorPoolCreateInfo(
            maxSets        = 1,
            poolSizeCount  = 1,
            pPoolSizes     = [pool_size],
        )
        self._dp = vk.vkCreateDescriptorPool(self._dev, pool_info, None)

        # Allocate set
        alloc_info = vk.VkDescriptorSetAllocateInfo(
            descriptorPool     = self._dp,
            descriptorSetCount = 1,
            pSetLayouts        = [self._dsl],
        )
        self._ds = vk.vkAllocateDescriptorSets(self._dev, alloc_info)[0]

        # Write descriptors
        bufs_mems = [
            (self._tgt_buf, 32),
            (self._cs_buf,  len(self.charset) * 4),
            (self._W_buf,   self.BATCH_MAX * 4),
            (self._wv_buf,  self.BATCH_MAX * 16 * 4),
            (self._mm_buf,  self.BATCH_MAX * 2  * 4),
        ]
        writes = []
        for i, (buf, sz) in enumerate(bufs_mems):
            buf_info = vk.VkDescriptorBufferInfo(buffer=buf, offset=0, range=sz)
            writes.append(vk.VkWriteDescriptorSet(
                dstSet          = self._ds,
                dstBinding      = i,
                descriptorCount = 1,
                descriptorType  = vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                pBufferInfo     = [buf_info],
            ))
        vk.vkUpdateDescriptorSets(self._dev, len(writes), writes, 0, None)

    # ── Command buffer ────────────────────────────────────────────────

    def _alloc_command_buffer(self):
        vk = self._vk
        cp_info = vk.VkCommandPoolCreateInfo(
            flags            = vk.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            queueFamilyIndex = self._qfam,
        )
        self._cmdpool = vk.vkCreateCommandPool(self._dev, cp_info, None)
        cb_info = vk.VkCommandBufferAllocateInfo(
            commandPool        = self._cmdpool,
            level              = vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount = 1,
        )
        self._cmdbuf = vk.vkAllocateCommandBuffers(self._dev, cb_info)[0]

        fence_info = vk.VkFenceCreateInfo()
        self._fence = vk.vkCreateFence(self._dev, fence_info, None)

    # ── Dispatch ─────────────────────────────────────────────────────

    def _write_target(self, target_bytes: bytes):
        """Upload SHA256 hash to GPU buffer.
        GPU reads uint32 as little-endian, but hash nibbles are big-endian.
        Byte-swap each 32-bit word so GPU reads correct big-endian values.
        e.g. bytes [a5,22,7e,61] need to be written as [61,7e,22,a5]
             so GPU reads uint32 = 0xa5227e61 (matches Python hash_hex nibbles)
        """
        vk = self._vk
        words   = struct.unpack('>8I', target_bytes)   # big-endian interpretation
        le_data = struct.pack('<8I', *words)            # byte-swap for little-endian GPU
        buf = vk.vkMapMemory(self._dev, self._tgt_mem, 0, 32, 0)
        buf[0:32] = le_data
        vk.vkUnmapMemory(self._dev, self._tgt_mem)

    def query_batch(self, target_bytes: bytes, batch_offset: int, n: int):
        """
        Run one GPU dispatch: n threads, positions (chain_len-1-batch_offset) .. down.
        Returns (W[n], wv[n,16], mm[n,2]) as numpy uint32 arrays.
        """
        vk = self._vk
        n = min(n, self.BATCH_MAX)
        groups = (n + 63) // 64  # ceil(n/64) workgroups of 64

        self._write_target(target_bytes)

        # Record command buffer
        vk.vkResetCommandBuffer(self._cmdbuf, 0)
        begin = vk.VkCommandBufferBeginInfo(
            flags=vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
        )
        vk.vkBeginCommandBuffer(self._cmdbuf, begin)

        vk.vkCmdBindPipeline(self._cmdbuf,
                             vk.VK_PIPELINE_BIND_POINT_COMPUTE, self._pipeline)
        vk.vkCmdBindDescriptorSets(self._cmdbuf,
                                   vk.VK_PIPELINE_BIND_POINT_COMPUTE,
                                   self._pl, 0, 1, [self._ds], 0, None)

        # Push constants: [chain_len, batch_offset, cs_len, str_len]
        # pValues is void* — must be cffi buffer, not raw Python bytes
        from vulkan._vulkan import ffi as _ffi
        pc_data = struct.pack('4i',
                              self.chain_len, batch_offset,
                              len(self.charset), self.str_len)
        pc_ptr = _ffi.from_buffer(pc_data)
        vk.vkCmdPushConstants(self._cmdbuf, self._pl,
                              vk.VK_SHADER_STAGE_COMPUTE_BIT,
                              0, 16, pc_ptr)

        vk.vkCmdDispatch(self._cmdbuf, groups, 1, 1)
        vk.vkEndCommandBuffer(self._cmdbuf)

        # Submit + wait
        vk.vkResetFences(self._dev, 1, [self._fence])
        submit = vk.VkSubmitInfo(
            commandBufferCount = 1,
            pCommandBuffers    = [self._cmdbuf],
        )
        vk.vkQueueSubmit(self._queue, 1, submit, self._fence)
        vk.vkWaitForFences(self._dev, 1, [self._fence],
                           vk.VK_TRUE, int(60e9))  # 60s timeout

        # Read back
        def read_buf(mem, count):
            buf = vk.vkMapMemory(self._dev, mem, 0, count * 4, 0)
            # vkMapMemory returns a cffi buffer object directly — use bytes() on it
            data = bytes(buf)[:count * 4]
            vk.vkUnmapMemory(self._dev, mem)
            return np.frombuffer(data, dtype=np.uint32).copy()

        out_W  = read_buf(self._W_mem,  n)
        out_wv = read_buf(self._wv_mem, n * 16).reshape(n, 16)
        out_mm = read_buf(self._mm_mem, n * 2).reshape(n, 2)

        return out_W, out_wv, out_mm

    # ── Cleanup ──────────────────────────────────────────────────────

    def destroy(self):
        if self._destroyed:
            return
        vk = self._vk
        vk.vkDeviceWaitIdle(self._dev)
        vk.vkDestroyFence(self._dev, self._fence, None)
        vk.vkFreeCommandBuffers(self._dev, self._cmdpool, 1, [self._cmdbuf])
        vk.vkDestroyCommandPool(self._dev, self._cmdpool, None)
        vk.vkDestroyDescriptorPool(self._dev, self._dp, None)
        vk.vkDestroyPipeline(self._dev, self._pipeline, None)
        vk.vkDestroyPipelineLayout(self._dev, self._pl, None)
        vk.vkDestroyDescriptorSetLayout(self._dev, self._dsl, None)
        vk.vkDestroyShaderModule(self._dev, self._shader_mod, None)
        for buf, mem in [
            (self._tgt_buf, self._tgt_mem),
            (self._cs_buf,  self._cs_mem),
            (self._W_buf,   self._W_mem),
            (self._wv_buf,  self._wv_mem),
            (self._mm_buf,  self._mm_mem),
        ]:
            vk.vkDestroyBuffer(self._dev, buf, None)
            vk.vkFreeMemory(self._dev, mem, None)
        vk.vkDestroyDevice(self._dev, None)
        vk.vkDestroyInstance(self._instance, None)
        self._destroyed = True

    def __enter__(self): return self
    def __exit__(self, *_): self.destroy()


import struct  # ensure available at module level


# ─────────────────────────────────────────────────────────────────────────────
# VulkanVerifyEngine — GPU forward chain walk + SHA256 comparison
#
# Replaces Python _verify_hit loop.  All candidates run in parallel.
#
# For each candidate (start_str, pos):
#   cur = start_str
#   for step in 0..pos-1: cur = cdp_reduce_inj(sha256(cur), step)
#   if sha256(cur) == target_hash: found
#
# Performance vs Python (110 verify/s):
#   9,016 candidates × avg 15,000 steps = 135M SHA256 ops
#   GPU: all threads parallel → ~<1s vs ~82s Python
# ─────────────────────────────────────────────────────────────────────────────

class VulkanVerifyEngine:
    """GPU-accelerated verify: all candidates in parallel, <1s vs 82s Python."""

    BATCH_MAX = 65536

    def __init__(self, charset: str, str_len: int,
                 device_index: int = 0, verbose: bool = True):
        import vulkan as vk
        self._vk       = vk
        self.charset   = charset
        self.str_len   = str_len
        self._destroyed = False

        # Load verify shader from same directory
        shader_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'cdp_verify.comp')
        glsl = open(shader_path, encoding='utf-8').read()
        if verbose:
            print("  [Vulkan Verify] Compiling verify shader (ACO)...")
        self._spirv = _compile_glsl_to_spirv(glsl)

        self._setup(device_index)
        self._build_pipeline()
        self._alloc_buffers()
        self._build_descriptor_set()
        self._alloc_command_buffer()

        if verbose:
            props = vk.vkGetPhysicalDeviceProperties(self._pdev)
            dn    = props.deviceName
            name  = dn.rstrip('\x00') if isinstance(dn, str) \
                    else bytes(dn).rstrip(b'\x00').decode()
            print(f"  [Vulkan Verify] Ready on '{name}'")

    # ── Vulkan boilerplate (identical to VulkanQueryEngine) ───────────────

    def _setup(self, preferred_index: int):
        vk = self._vk
        app_info  = vk.VkApplicationInfo(
            pApplicationName   = b"cdp_vulkan_verify",
            applicationVersion = vk.VK_MAKE_VERSION(1, 0, 0),
            pEngineName        = b"none",
            engineVersion      = vk.VK_MAKE_VERSION(1, 0, 0),
            apiVersion         = vk.VK_API_VERSION_1_0,
        )
        self._instance = vk.vkCreateInstance(
            vk.VkInstanceCreateInfo(pApplicationInfo=app_info), None)

        pdevs = vk.vkEnumeratePhysicalDevices(self._instance)
        if not pdevs:
            raise RuntimeError("[Vulkan Verify] No physical devices found")
        pdevs_sorted = sorted(
            pdevs,
            key=lambda pd: 0 if vk.vkGetPhysicalDeviceProperties(pd).vendorID == 0x1002 else 1)
        self._pdev = pdevs_sorted[min(preferred_index, len(pdevs_sorted)-1)]

        qprops = vk.vkGetPhysicalDeviceQueueFamilyProperties(self._pdev)
        self._qfam = next(
            (i for i, qp in enumerate(qprops)
             if qp.queueFlags & vk.VK_QUEUE_COMPUTE_BIT), None)
        if self._qfam is None:
            raise RuntimeError("[Vulkan Verify] No compute queue family")

        queue_info = vk.VkDeviceQueueCreateInfo(
            queueFamilyIndex=self._qfam, queueCount=1, pQueuePriorities=[1.0])
        self._dev   = vk.vkCreateDevice(
            self._pdev,
            vk.VkDeviceCreateInfo(pQueueCreateInfos=[queue_info],
                                  queueCreateInfoCount=1,
                                  enabledExtensionCount=0), None)
        self._queue = vk.vkGetDeviceQueue(self._dev, self._qfam, 0)

        mem_props = vk.vkGetPhysicalDeviceMemoryProperties(self._pdev)
        self._mem_host_idx = self._find_memory_type(
            mem_props,
            vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)

    def _find_memory_type(self, mem_props, flags) -> int:
        for i in range(mem_props.memoryTypeCount):
            if (mem_props.memoryTypes[i].propertyFlags & flags) == flags:
                return i
        raise RuntimeError(f"[Vulkan Verify] No memory type for flags={flags:#x}")

    def _make_buffer(self, size_bytes: int):
        vk = self._vk
        buf = vk.vkCreateBuffer(self._dev,
            vk.VkBufferCreateInfo(
                size=size_bytes,
                usage=vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE), None)
        req  = vk.vkGetBufferMemoryRequirements(self._dev, buf)
        mem  = vk.vkAllocateMemory(self._dev,
            vk.VkMemoryAllocateInfo(
                allocationSize=req.size,
                memoryTypeIndex=self._mem_host_idx), None)
        vk.vkBindBufferMemory(self._dev, buf, mem, 0)
        return buf, mem

    def _build_pipeline(self):
        vk = self._vk
        mod_info = vk.VkShaderModuleCreateInfo(
            codeSize=len(self._spirv), pCode=bytes(self._spirv))
        self._shader_mod = vk.vkCreateShaderModule(self._dev, mod_info, None)

        # 6 bindings: starts, positions, target, charset, found, results
        bindings = [
            vk.VkDescriptorSetLayoutBinding(
                binding=i,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=1,
                stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT)
            for i in range(6)
        ]
        self._dsl = vk.vkCreateDescriptorSetLayout(self._dev,
            vk.VkDescriptorSetLayoutCreateInfo(
                bindingCount=len(bindings), pBindings=bindings), None)

        pc_range = vk.VkPushConstantRange(
            stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT, offset=0, size=16)
        self._pl = vk.vkCreatePipelineLayout(self._dev,
            vk.VkPipelineLayoutCreateInfo(
                setLayoutCount=1, pSetLayouts=[self._dsl],
                pushConstantRangeCount=1, pPushConstantRanges=[pc_range]), None)

        stage = vk.VkPipelineShaderStageCreateInfo(
            stage=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            module=self._shader_mod, pName=b"main")
        self._pipeline = vk.vkCreateComputePipelines(self._dev, None, 1,
            [vk.VkComputePipelineCreateInfo(stage=stage, layout=self._pl)],
            None)[0]

    def _alloc_buffers(self):
        N  = self.BATCH_MAX
        SL = self.str_len
        cs = self.charset.encode()

        self._starts_buf, self._starts_mem = self._make_buffer(N * SL * 4)
        self._pos_buf,    self._pos_mem    = self._make_buffer(N * 4)
        self._tgt_buf,    self._tgt_mem    = self._make_buffer(32)
        self._cs_buf,     self._cs_mem     = self._make_buffer(len(cs) * 4)
        self._found_buf,  self._found_mem  = self._make_buffer(N * 4)
        self._res_buf,    self._res_mem    = self._make_buffer(N * SL * 4)

        # Upload charset once
        data = struct.pack(f'{len(cs)}I', *cs)
        buf  = self._vk.vkMapMemory(self._dev, self._cs_mem, 0, len(data), 0)
        buf[0:len(data)] = data
        self._vk.vkUnmapMemory(self._dev, self._cs_mem)

    def _build_descriptor_set(self):
        vk = self._vk
        N  = self.BATCH_MAX
        SL = self.str_len
        cs_size = len(self.charset) * 4

        pool_size = vk.VkDescriptorPoolSize(
            type=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descriptorCount=6)
        self._dp = vk.vkCreateDescriptorPool(self._dev,
            vk.VkDescriptorPoolCreateInfo(
                maxSets=1, poolSizeCount=1, pPoolSizes=[pool_size]), None)
        self._ds = vk.vkAllocateDescriptorSets(self._dev,
            vk.VkDescriptorSetAllocateInfo(
                descriptorPool=self._dp,
                descriptorSetCount=1,
                pSetLayouts=[self._dsl]))[0]

        bufs_sizes = [
            (self._starts_buf, N * SL * 4),
            (self._pos_buf,    N * 4),
            (self._tgt_buf,    32),
            (self._cs_buf,     cs_size),
            (self._found_buf,  N * 4),
            (self._res_buf,    N * SL * 4),
        ]
        writes = []
        for i, (buf, sz) in enumerate(bufs_sizes):
            buf_info = vk.VkDescriptorBufferInfo(buffer=buf, offset=0, range=sz)
            writes.append(vk.VkWriteDescriptorSet(
                dstSet=self._ds, dstBinding=i,
                descriptorCount=1,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                pBufferInfo=[buf_info]))
        vk.vkUpdateDescriptorSets(self._dev, len(writes), writes, 0, None)

    def _alloc_command_buffer(self):
        vk = self._vk
        self._cmdpool = vk.vkCreateCommandPool(self._dev,
            vk.VkCommandPoolCreateInfo(
                flags=vk.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
                queueFamilyIndex=self._qfam), None)
        self._cmdbuf = vk.vkAllocateCommandBuffers(self._dev,
            vk.VkCommandBufferAllocateInfo(
                commandPool=self._cmdpool,
                level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                commandBufferCount=1))[0]
        self._fence = vk.vkCreateFence(self._dev, vk.VkFenceCreateInfo(), None)

    # ── Core verify method ────────────────────────────────────────────────

    def verify_batch(self, candidates, target_bytes: bytes):
        """
        Verify all (start_str, pos) candidates against target_bytes in parallel.

        candidates: list of (pos: int, start_str: str)  — sorted by pos for better wavefront efficiency
        target_bytes: 32-byte SHA256 hash

        Returns: first matching plaintext string, or None.
        """
        vk  = self._vk
        SL  = self.str_len
        cs  = len(self.charset)
        n   = min(len(candidates), self.BATCH_MAX)

        # Pack starts (uint32 per char) and positions
        starts_data = []
        pos_data    = []
        for pos_val, start_str in candidates[:n]:
            for ch in start_str:
                starts_data.append(ord(ch))
            pos_data.append(pos_val)

        # Upload starts
        raw = struct.pack(f'{len(starts_data)}I', *starts_data)
        buf = vk.vkMapMemory(self._dev, self._starts_mem, 0, len(raw), 0)
        buf[0:len(raw)] = raw
        vk.vkUnmapMemory(self._dev, self._starts_mem)

        # Upload positions
        raw = struct.pack(f'{n}I', *pos_data)
        buf = vk.vkMapMemory(self._dev, self._pos_mem, 0, len(raw), 0)
        buf[0:len(raw)] = raw
        vk.vkUnmapMemory(self._dev, self._pos_mem)

        # Upload target hash (big-endian → little-endian for GPU)
        words   = struct.unpack('>8I', target_bytes)
        le_data = struct.pack('<8I', *words)
        buf = vk.vkMapMemory(self._dev, self._tgt_mem, 0, 32, 0)
        buf[0:32] = le_data
        vk.vkUnmapMemory(self._dev, self._tgt_mem)

        # Clear found flags
        zero = struct.pack(f'{n}I', *([0] * n))
        buf = vk.vkMapMemory(self._dev, self._found_mem, 0, n * 4, 0)
        buf[0:n * 4] = zero
        vk.vkUnmapMemory(self._dev, self._found_mem)

        # Record + dispatch
        groups = (n + 63) // 64
        vk.vkResetCommandBuffer(self._cmdbuf, 0)
        vk.vkBeginCommandBuffer(self._cmdbuf,
            vk.VkCommandBufferBeginInfo(
                flags=vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT))
        vk.vkCmdBindPipeline(
            self._cmdbuf, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self._pipeline)
        vk.vkCmdBindDescriptorSets(
            self._cmdbuf, vk.VK_PIPELINE_BIND_POINT_COMPUTE,
            self._pl, 0, 1, [self._ds], 0, None)

        from vulkan._vulkan import ffi as _ffi
        pc_data = struct.pack('4i', n, cs, SL, 0)
        pc_ptr  = _ffi.from_buffer(pc_data)
        vk.vkCmdPushConstants(
            self._cmdbuf, self._pl,
            vk.VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, pc_ptr)

        vk.vkCmdDispatch(self._cmdbuf, groups, 1, 1)
        vk.vkEndCommandBuffer(self._cmdbuf)

        vk.vkResetFences(self._dev, 1, [self._fence])
        vk.vkQueueSubmit(self._queue, 1,
            vk.VkSubmitInfo(
                commandBufferCount=1, pCommandBuffers=[self._cmdbuf]),
            self._fence)
        vk.vkWaitForFences(self._dev, 1, [self._fence], vk.VK_TRUE, int(60e9))

        # Read back found flags
        raw_found = bytes(vk.vkMapMemory(self._dev, self._found_mem, 0, n*4, 0))[:n*4]
        vk.vkUnmapMemory(self._dev, self._found_mem)
        flags = struct.unpack(f'{n}I', raw_found)

        # Read back results for any found candidate
        raw_res = bytes(vk.vkMapMemory(self._dev, self._res_mem, 0, n*SL*4, 0))[:n*SL*4]
        vk.vkUnmapMemory(self._dev, self._res_mem)
        res_uints = struct.unpack(f'{n*SL}I', raw_res)

        for i, flag in enumerate(flags):
            if flag:
                plaintext = ''.join(
                    chr(res_uints[i * SL + j]) for j in range(SL))
                return plaintext

        return None

    # ── Cleanup ──────────────────────────────────────────────────────────

    def destroy(self):
        if self._destroyed:
            return
        vk = self._vk
        vk.vkDeviceWaitIdle(self._dev)
        vk.vkDestroyFence(self._dev, self._fence, None)
        vk.vkFreeCommandBuffers(self._dev, self._cmdpool, 1, [self._cmdbuf])
        vk.vkDestroyCommandPool(self._dev, self._cmdpool, None)
        vk.vkDestroyDescriptorPool(self._dev, self._dp, None)
        vk.vkDestroyPipeline(self._dev, self._pipeline, None)
        vk.vkDestroyPipelineLayout(self._dev, self._pl, None)
        vk.vkDestroyDescriptorSetLayout(self._dev, self._dsl, None)
        vk.vkDestroyShaderModule(self._dev, self._shader_mod, None)
        for buf, mem in [
            (self._starts_buf, self._starts_mem),
            (self._pos_buf,    self._pos_mem),
            (self._tgt_buf,    self._tgt_mem),
            (self._cs_buf,     self._cs_mem),
            (self._found_buf,  self._found_mem),
            (self._res_buf,    self._res_mem),
        ]:
            vk.vkDestroyBuffer(self._dev, buf, None)
            vk.vkFreeMemory(self._dev, mem, None)
        vk.vkDestroyDevice(self._dev, None)
        vk.vkDestroyInstance(self._instance, None)
        self._destroyed = True

    def __enter__(self): return self
    def __exit__(self, *_): self.destroy()

