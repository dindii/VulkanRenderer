#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/vec4.hpp>
#include <glm/mat4x4.hpp>

#include <iostream>
#include <vector>
#include <string>

#include <cstdint> // Necessary for uint32_t
#include <limits> // Necessary for std::numeric_limits
#include <algorithm> // Necessary for std::clamp

#include <array>

#include <fstream>

#include <map>

const uint32_t APP_WIDTH = 2560;
const uint32_t APP_HEIGHT = 1440;
const uint32_t MAX_FRAMES_IN_FLIGHT = 2;

static uint32_t s_CurrentFrame = 0;

const char* g_RequiredValidationLayer = "VK_LAYER_KHRONOS_validation";

struct Vertex
{
	glm::vec2 position;
	glm::vec3 color;

	static VkVertexInputBindingDescription GetBindingDescription()
	{
		VkVertexInputBindingDescription bindingDescription = {};

		//This concept of binding we have on Vulkan is basically where the Vertex shader should begin to read data from
		//so, for instance, let's say you have really a lot of interleaved data, such as POS-UV-NORMAL POS-UV-NORMAL POS-UV-NORMAL
		//we are going to read this data from the beginning of the buffer and we know how far away we are from the next vertex
		//so we only need one binding, this is, one location to start from: the beginning. 
		//Regarding to locations, is just a matter of mapping the attributes to the correct shader variables. In this case we will have 3 locations (each one has its own stride)
		//If we have, for instance, a vertex buffer with all the positions first, then all the UVs and by last all the normals,
		//then we would have 3 bindings to the same vertex buffer because we would read to different parts of the buffer
		//The locations would be the same, after all we still have 3 attributes and need to assign shader variables.
		//The stride would be the same, but instead of setting the offset for each attribute (begin 0 to nPos for pos, then nPos to UV for beginning of UV etc)...
		//we define this in an offset array that describes after how many bytes other attribute begins.
		///more: https://docs.vulkan.org/guide/latest/vertex_input_data_processing.html
		bindingDescription.binding = 0;
		bindingDescription.stride = sizeof(Vertex);
		bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		return bindingDescription;
	}

	static std::array<VkVertexInputAttributeDescription, 2> GetAttributeDescriptions()
	{
		std::array<VkVertexInputAttributeDescription, 2> attributeDescriptions = {};

		
		attributeDescriptions[0].binding = 0;
		//Location is the actual Location (Location = 0 etc) inside the shader
		attributeDescriptions[0].location = 0;
		attributeDescriptions[0].format = VK_FORMAT_R32G32_SFLOAT;
		//Just like attribute layouts on OPENGL.
		attributeDescriptions[0].offset = offsetof(Vertex, position);

		attributeDescriptions[1].binding = 0;
		attributeDescriptions[1].location = 1;
		attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[1].offset = offsetof(Vertex, color);

		return attributeDescriptions;
	}
};

static const std::vector<Vertex> s_Vertices = {
	{{0.0f, -0.5f}, {1.0f, 0.0f, 0.0f}},
	{{0.5f, 0.5f}, {1.0f, 0.0f, 1.0f}},
	{{-0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}} 
};

template<typename T>
inline T clamp(const T min, const T max, const T val)
{
	T aux = 0;

	if (val < min)
	{
		aux = min;
		return aux;
	}
	else if (val > max)
	{
		aux = max;
		return aux;
	}

	return val;
}

static std::vector<char> ReadFile(const std::string& fileName)
{
	//The advantage of starting to read at the end of the file is that we can use the read position to determine the size of the file and allocate a buffer:
	//also, binary to avoid text transformations ('\n' etc)
	std::ifstream file(fileName, std::ios::ate | std::ios::binary);

	if (!file.is_open())
	{
		std::cerr << "Failed to open file! " << fileName << std::endl;
		__debugbreak();
	}

	size_t fileSize = (size_t)file.tellg();
	std::vector<char> buffer(fileSize);

	//then lets get back to the beginning of the file and read everything all at once
	file.seekg(0);
	file.read(buffer.data(), fileSize);

	file.close();

	return buffer;
}

void VkCheck(VkResult res)
{
	if (res != VK_SUCCESS)
	{
		std::cerr << "Unsuccessful Vulkan Operation!" << std::endl;
		__debugbreak();
	}
}

#ifdef VKDEBUG
VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger)
{
	auto loadedFunc = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");

	if (loadedFunc != nullptr)
	{
		return loadedFunc(instance, pCreateInfo, pAllocator, pDebugMessenger);
	}
	else
	{
		return VK_ERROR_EXTENSION_NOT_PRESENT;
	}

}

void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator)
{
	auto loadedFunc = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");

	if (loadedFunc != nullptr)
		loadedFunc(instance, debugMessenger, pAllocator);
}
#endif

struct SwapChainSupportDetails
{
	//Num of backbuffers, width, height etc..
	VkSurfaceCapabilitiesKHR capabilities;

	//Formats to present (rgba, rgb, 16, 32 bits etc..)
	std::vector<VkSurfaceFormatKHR> formats;

	std::vector<VkPresentModeKHR> presentModes;
};


class HelloTriangleApp
{
public:
	void Run()
	{
		InitWindow();
		InitVulkan();

#ifdef VKDEBUG
		//Now that the instance is created, lets create, fill and attach our m_DebugMessenger to this instance
		//So that way we are going to get error messages regarding the instance, such as about objects made with it.
		//The instance's internal messenger will still be running anyways. It is more useful to get Create/Destroy instance issues.
		SetupDebugMessenger();
#endif

		CreateWindowSurface();

		//Retrieve a physical GPU handle. (we will use it to create a Logical handle then)
		PickPhysicalDevice();

		//Use the physical GPU to create a logical handle to it
		//we will use this handle to access GPU functions (e.g: submit commands to queue, dispatch queue etc)
		CreateGPULogicalHandle();

		CreateSwapChain();
	    CreateSwapChainImageViews();

		CreateRenderPass();
		CreateGraphicsPipeline();
		CreateFramebuffers();
		CreateCommandPool();
		CreateVertexBuffer();
		CreateCommandBuffers();
		CreateSyncObjects();


		Update();
		Destroy();
	};

private:

	void CreateBuffer(VkDeviceSize size, VkBufferUsageFlags usageFlags, VkMemoryPropertyFlags propertiesFlags, VkBuffer& buffer, VkDeviceMemory& bufferMemory)
	{
		VkBufferCreateInfo bufferInfo = {};
		bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		bufferInfo.size = size;
		bufferInfo.usage = usageFlags;
		bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		
		VkCheck(vkCreateBuffer(m_Device, &bufferInfo, nullptr, &buffer));
		
		VkMemoryRequirements memReq;
		vkGetBufferMemoryRequirements(m_Device, buffer, &memReq);
		
		VkMemoryAllocateInfo allocInfo = {};
		allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocInfo.allocationSize = memReq.size;
		allocInfo.memoryTypeIndex = FindMemoryType(memReq.memoryTypeBits, propertiesFlags);

		VkCheck(vkAllocateMemory(m_Device, &allocInfo, nullptr, &bufferMemory));

		VkCheck(vkBindBufferMemory(m_Device, buffer, bufferMemory, 0));

		/* Old Comments about buffer/memory allocation using vertex buffer as an example: 
				//Before creating the buffer, we must specify, as ever, a info struct.
		//we will specify the size, how we are going to use it and based on that, it will specify some musts the device need to have
		//in order to this buffer to be created and used correctly.
		//One thing to mention is that, creating the buffer itself is not the same as allocating the memory.
		//First we will specify the buffer. Then we will retrieve what it needs to be created
		//then we allocate actual memory in the GPU with those specifications
		//and then we assign (bind) that buffer to that space allocated. So the data will be inside the buffer, inside the space.
		VkBufferCreateInfo bufferInfo = {};
		bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		bufferInfo.size = sizeof(s_Vertices[0]) * s_Vertices.size();
		bufferInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT; //we are going to use it as a vertex buffer
		bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE; //only one family queue will be using that. The graphics one.
		bufferInfo.flags = 0;

		//Create the buffer
		VkCheck(vkCreateBuffer(m_Device, &bufferInfo, nullptr, &m_VertexBuffer));

		//Now that the buffer is created, we need to fetch a bunch of properties that this buffer needs
		//so we will use that to allocate space that suits this buffer well
		//those properties will be mainly MemoryType and Flags (if it is visible from the CPU/Host etc).
		//The MemoryType mainly specifies how a memory will be allocated, so we need to find the right one for our buffer needs
		VkMemoryRequirements memRequirements;
		vkGetBufferMemoryRequirements(m_Device, m_VertexBuffer, &memRequirements);

		//We also need to fill a info struct that defines how we are going to actual allocate the memory for the buffer
		VkMemoryAllocateInfo allocInfo = {};
		allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;

		//For different types of buffer, Vulkan may need more memory than the actual data. Maybe because of alignment, maybe because some header of control block etc
		allocInfo.allocationSize = memRequirements.size;

		//Find the best way to allocate this memory. We will simply go through all the memory types our device have and find one that is both CPU visible and host coherent
		//we will take the first one that supports both and use that to allocate our memory
		allocInfo.memoryTypeIndex = FindMemoryType(memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

		//Allocate the memory and set the handle
		VkCheck(vkAllocateMemory(m_Device, &allocInfo, nullptr, &m_VertexBufferMemory));

		//Bind the Buffer with the actual created memory
		//the buffer is like a view of how the memory should be allocated, laid out etc
		vkBindBufferMemory(m_Device, m_VertexBuffer, m_VertexBufferMemory, 0);
		*/
	}

	void CreateVertexBuffer()
	{
		VkDeviceSize bufferSize = sizeof(s_Vertices[0]) * s_Vertices.size();

		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMemory;
		CreateBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

		//With the VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT flag we used, we can then map GPU memory to CPU memory
		//we will simply ask for a region of memory to me mapped and pass a pointer. We will get this pointer pointing to the mapped region
		//we also could just pass VK_WHOLE_SIZE instead of size, to map to the whole chunk
		//This is not the best approach for performance (such as staging buffers) but it is useful for many small purposes 
		void* data;
		vkMapMemory(m_Device, stagingBufferMemory, 0, bufferSize, 0, &data);
		//copy all the data from the vertex buffer cpu to the gpu mapped memory
		memcpy(data, s_Vertices.data(), bufferSize);
		//and then unmap
		vkUnmapMemory(m_Device, stagingBufferMemory);

		//Usually, memory transfer doesn't need to happen immediately because of N factors (cache is one of them)
		//also, sometimes even that the transfer occurred, it is not still visible that this happened
		//so we need to flush that. But since we specify the flag VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, the driver is well aware of that transfer
		//and will invalidate the cache as soon as the transfer is made

		CreateBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, m_VertexBuffer, m_VertexBufferMemory);

		CopyBuffer(stagingBuffer, m_VertexBuffer, bufferSize);

		vkDestroyBuffer(m_Device, stagingBuffer, nullptr);
		vkFreeMemory(m_Device, stagingBufferMemory, nullptr);
	}

	uint32_t FindMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties)
	{
		VkPhysicalDeviceMemoryProperties memProps;
		vkGetPhysicalDeviceMemoryProperties(m_PhysicalGPU, &memProps);

		
		for (uint32_t i = 0; i < memProps.memoryTypeCount; i++)
		{
			//We simply run through all of the memory types that our device have and once we find any that we support (bit set)
			//we check also if we support the properties we want, in this case, to be visible from CPU and to be cache coherent
			//typeFilter is a memRequirements.memoryTypeBits bitfield, it is guaranteed by Vulkan API that the index of the bit in the bitfield
			//corresponds to the index of the memProps.memoryTypes array.
			//So if the 2nd bit is set on the typeFilter, then the property array index that corresponds to that memory type is 2.
			if ((typeFilter & (1 << i)) && (memProps.memoryTypes[i].propertyFlags & properties) == properties)
				return i;
		}

	}

	void CopyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size)
	{
		VkCommandBufferAllocateInfo allocInfo = {};
		allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		allocInfo.commandPool = m_TemporaryCommandPool;
		allocInfo.commandBufferCount = 1;

		VkCommandBuffer	tempCommandBuffer;
		vkAllocateCommandBuffers(m_Device, &allocInfo, &tempCommandBuffer);

		VkCommandBufferBeginInfo beginInfo = {};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

		//We are only going to use that command buffer once, this is, just to copy memory and exit.
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

		vkBeginCommandBuffer(tempCommandBuffer, &beginInfo);

		VkBufferCopy copyRegion = {};
		copyRegion.srcOffset = 0;
		copyRegion.dstOffset = 0;
		copyRegion.size = size;

		//We can copy more than one region, we could pass an array of VkBufferCopy 
		vkCmdCopyBuffer(tempCommandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

		vkEndCommandBuffer(tempCommandBuffer);


		VkSubmitInfo submitInfo = {};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &tempCommandBuffer;

		VkFenceCreateInfo memoryTransferredFenceInfo = {};
		memoryTransferredFenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		VkFence memoryTransferredFence;

		vkCreateFence(m_Device, &memoryTransferredFenceInfo, nullptr, &memoryTransferredFence);

		vkQueueSubmit(m_GraphicsQueueHandle, 1, &submitInfo, memoryTransferredFence);

		//We submit our memory transfer command and then we just freeze the CPU waiting to the memory to be transferred
		//we then use this allocated memory to draw in the future. So we will wait all previous commands to reach this point, and then we can continue to work from here. 
		//That way we don't have any race condition and commands from now will be updated.
		//we also could use VkQueueWaitIdle with the graphics queue. The API documentation says:
		/*"vkQueueWaitIdle is equivalent to having submitted a valid fence to every previously executed queue submission command that accepts a fence, 
		then waiting for all of those fences to signal using vkWaitForFences with an infinite timeout and waitAll set to VK_TRUE."*/
		//and this is equivalent to use the fence and block the CPU until it has reached that point (where all our command list/copy commands are finished).
		vkWaitForFences(m_Device, 1, &memoryTransferredFence, VK_TRUE, UINT64_MAX);

		// -- 

		vkDestroyFence(m_Device, memoryTransferredFence, nullptr);
		vkFreeCommandBuffers(m_Device, m_TemporaryCommandPool, 1, &tempCommandBuffer);
	}

	void CreateSyncObjects()
	{
		m_RenderTargetAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
		m_RenderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
		m_InFlightFrameFences.resize(MAX_FRAMES_IN_FLIGHT);

		VkSemaphoreCreateInfo semaphoreInfo = {};
		semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

		VkFenceCreateInfo fenceInfo = {};
		fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

		for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
		{
			VkCheck(vkCreateSemaphore(m_Device, &semaphoreInfo, nullptr, &m_RenderTargetAvailableSemaphores[i]));
			VkCheck(vkCreateSemaphore(m_Device, &semaphoreInfo, nullptr, &m_RenderFinishedSemaphores[i]));
			VkCheck(vkCreateFence(m_Device, &fenceInfo, nullptr, &m_InFlightFrameFences[i]));
		}
	}


	void RecordCommands(VkCommandBuffer buffer, uint32_t imageIndex)
	{
		VkCommandBufferBeginInfo beginInfo = {};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = 0;
		beginInfo.pInheritanceInfo = nullptr;

		VkCheck(vkBeginCommandBuffer(buffer, &beginInfo));

		VkRenderPassBeginInfo renderPassInfo = {};
		renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
		renderPassInfo.renderPass = m_RenderPass;
		renderPassInfo.framebuffer = m_SwapChainFramebuffers[imageIndex];
		
		//RenderArea is basically on what pixels we will be running our shaders and loading/storing the result onto.
		//Best perf when matching with attachments size.
		renderPassInfo.renderArea.offset = { 0, 0 };
		renderPassInfo.renderArea.extent = m_RenderTargetExtent;

		VkClearValue clearColor = {};
		clearColor.color = { 0.0f, 0.0f, 0.0f, 1.0f };
		renderPassInfo.clearValueCount = 1;
		renderPassInfo.pClearValues = &clearColor; //used for VK_ATTACHMENT_LOAD_OP_CLEAR

		//All functions with prefix vkCmd are commands that are being recorded into buffer
		//we are recording a command to begin render pass, another to bind pipeline, another to set viewport etc.
		vkCmdBeginRenderPass(buffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
		vkCmdBindPipeline(buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_GraphicsPipeline);
		
		VkBuffer vertexBuffersToBind[] = { m_VertexBuffer };

		//This is where we set the offset inside the vertex buffer. If we have, for instance, a buffer that is not interleaved, we would here set the offset for each bind (for instance, UV * UVElements).
		VkDeviceSize vertexBuffersToBindOffsets[] = { 0 };
		vkCmdBindVertexBuffers(buffer, 0, 1, vertexBuffersToBind, vertexBuffersToBindOffsets);

		VkViewport viewport = {};
		viewport.x = 0.0f;
		viewport.y = 0.0f;
		viewport.width = (float)m_RenderTargetExtent.width;
		viewport.height = (float)m_RenderTargetExtent.height;
		viewport.minDepth = 0.0f;
		viewport.maxDepth = 1.0f;
		vkCmdSetViewport(buffer, 0, 1, &viewport);

		VkRect2D scissor = {};
		scissor.offset = { 0, 0 };
		scissor.extent = m_RenderTargetExtent;
		vkCmdSetScissor(buffer, 0, 1, &scissor);

		vkCmdDraw(buffer, s_Vertices.size(), 1, 0, 0);

		vkCmdEndRenderPass(buffer);

		VkCheck(vkEndCommandBuffer(buffer));
	}

	void CreateCommandBuffers()
	{
		m_CommandBuffers.resize(MAX_FRAMES_IN_FLIGHT);

		VkCommandBufferAllocateInfo allocInfo = {};
		allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInfo.commandPool = m_CommandPool;

		//we have two levels: primary and secondary. Primary command buffers can be sent directly for the queue but
		//cannot be called from other command buffers.
		//Second buffers are mainly to be called from other command buffers and cannot be sent directly to the queue.
		//This is commonly used for operations that is common of multiple buffers.
		allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		allocInfo.commandBufferCount = (uint32_t)m_CommandBuffers.size();

		//If the command buffer was already recorded once, then a call to vkBeginCommandBuffer will implicitly reset it.
		VkCheck(vkAllocateCommandBuffers(m_Device, &allocInfo, m_CommandBuffers.data()));
	}

	void CreateCommandPool()
	{
		VkCommandPoolCreateInfo poolInfo = {};
		poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
		poolInfo.queueFamilyIndex = m_GraphicsQueueFamilyType;
		
		VkCheck(vkCreateCommandPool(m_Device, &poolInfo, nullptr, &m_CommandPool));



		VkCommandPoolCreateInfo temporaryPoolInfo = {};
		temporaryPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		//We will use this pool to create temporary staging buffers
		temporaryPoolInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
		temporaryPoolInfo.queueFamilyIndex = m_GraphicsQueueFamilyType;

		VkCheck(vkCreateCommandPool(m_Device, &temporaryPoolInfo, nullptr, &m_TemporaryCommandPool));
	}
	void CreateFramebuffers()
	{
		m_SwapChainFramebuffers.resize(m_RenderTargetViews.size());

		for (int i = 0; i < m_SwapChainFramebuffers.size(); i++)
		{
			VkImageView attachments[] = { m_RenderTargetViews[i] };

			VkFramebufferCreateInfo framebufferInfo = {};
			framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
			framebufferInfo.renderPass = m_RenderPass;
			framebufferInfo.attachmentCount = 1;

			//this is the attachments that we specify on renderpass
			framebufferInfo.pAttachments = attachments;
			framebufferInfo.width = m_RenderTargetExtent.width;
			framebufferInfo.height = m_RenderTargetExtent.height;
			framebufferInfo.layers = 1;

			VkCheck(vkCreateFramebuffer(m_Device, &framebufferInfo, nullptr, &m_SwapChainFramebuffers[i]));
		}

	}
	void CreateRenderPass()
	{
		//Lets create a render pass object
		//we will only have one attachment (color buffer) on our framebuffer. This attachment will be one of the swap chain images
		VkAttachmentDescription colorAttachment = {};
		//as the attachment itself is the swap chain image, let's use the same format so it can write to it correctly
		colorAttachment.format = m_RenderTargetFormat; 
		//usually this is used for multisampling. We are not doing anything with it just yet
		colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;

		//defines what we do with the color buffer before rendering. In this case we will just clear it so we don't have 
		//any remnants from the past draw. We could also clear it with a specific color, don't do anything or keep the remnants.
		//Also, the DONT_CARE is basically: I don't care about what values it has, I don't want to set it to any specific value, just use some fast clear algorithm for me.
		//Usually that's the case when you are not dealing with depth buffers (in which you do want to specify a value to clear).	
		colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		//This is after we draw to the buffer. We only have two options, to keep the image in memory to read later or to set it as undefined and don't do anything.
		//We also could DONT_CARE about it. And we say that it probably will never show up outside a render/subpass. Sounds good for internal ops
		colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		
		//We are not doing anything with the stencil buffer so we don't care about it. (the logic is same as the load/store op)
		colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		

		//VkImages (our attachments) have special memory layouts defined for each case of usage. In order to make a certain operation
		//we need to set the best and right layout that will support this operation.
		//For instance, after drawing to an image, we don't care about its format (because we are going to clear it anyway)
		//but after drawing to it, we must change its layout to PRESENT because we want to present this Image to the screen
		//and this layout is the most efficient (and expected) for such operation. Thus the image will be ready to presentation for the swap chain.
		//The Undefined value usually doesn't guarantee the image to be the same in the previous draw.
		//this is probably because when we set a Before state, it maybe can try to return the image to that Before state.
		//In our case that's ok because we don't keep the content of the previous draw as we just clear the image.
		colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

		VkAttachmentReference colorAttachmentRef = {};
		colorAttachmentRef.attachment = 0; //index 0 as we only have 1 attachment
		
		//Inside the render pass and subpass, we will use this image as an color buffer indeed. So the best option is to use it as optimial color attachment.
		//After everything then it will be transitioned to present_src_khr. 
		//This layout will be transitioned by Vulkan when the subpass begins.
		colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		VkSubpassDescription subpass = {};
		//specifies that this is a graphics subpass
		subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpass.colorAttachmentCount = 1;

		//those attachments can be referenced from the fragment shader using layout(location = 0) out vec4 outColor
		subpass.pColorAttachments = &colorAttachmentRef;

		//Here is where we are going to tie everything.
		//We have a bunch of Attachments (color, depth, stencil etc) in an array. And we will pass this array here.
		//Then we have the SubPasses, each subpass can have one or more attachment refs, those attachment refs does not directly refer to the attachment but to the index of the array on the render pass
		//So we have the renderpass with a bunch of images, and inside the renderpass we have a bunch of subpasses, each subpass declares which attachment on the renderpass it wants to use.
		//It does so by referencing an index. So we create an AttachmentRef that points to index 0. Then we add this AttachmentRef to a Subpass, then we add this Subpass to a Renderpass. It will seek for index 0 on the attachment arrays on the renderpass.
		//I can assume that this way, multiple renderpass can read/write to the same memory location without having to copy stuff around. Also, when you use multiple subpasses in a pipeline (let's say post processing) vulkan can sort this and optimize it.
		VkRenderPassCreateInfo renderPassInfo = {};
		renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		renderPassInfo.attachmentCount = 1;
		renderPassInfo.pAttachments = &colorAttachment;
		renderPassInfo.subpassCount = 1;
		renderPassInfo.pSubpasses = &subpass;

		//Now we will set dependencies between subpasses.

		VkSubpassDependency dependency = {};
		//We say that the source subpass (i.e the one we are waiting on) is something prior to this  pass
		//usually it can be any operation or set of operations before
		dependency.srcSubpass = VK_SUBPASS_EXTERNAL;

		//the pass that will wait will be this one (0 index, the only one we have)
		dependency.dstSubpass = 0;

		//all previous commands must reach at least this stage before the dst stages can proceed.
		//this tells Vulkan what we need to finish here first before moving on to the next subpass
		//because they are out of order and may be the operations within all subpasses.
		//so dst (us) will only proceed when the previous subpass (in our case, the previous renderpass )
		//have written to the texture (outputted to the attachment color of the framebuffer)
		dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		
		//This just assumes that when the image is outputted, the swapchain will read it and we are then ready to go
		//because when defining the submit info, we set a semaphore that will only be flagged when we have a new render target
		//and this semaphore will make the command list to wait on the OUTPUT_BIT
		//so wait on two conditions before writing to an image, one is to have an image and another is to make sure that the previous image was written


		//this effectively tells what stages we are not allowed to run on our subpass
		//until all stages referenced on srcStageMasks are done.
		//in this case, we are not allowed to write to an image until the previous renderpass has written to the image
		dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

		//we also specify what kind of access we are doing on each subass. In our case, the source subpass (prior to us)
		//are going to write to this render target.
		dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		
		//we are not doing anything other than presenting it at the swap chain (not using it in any shader or so)
		//so we don't need to specify anything, but if we were reading it from a shader, we would need to specify READ access.
		dependency.srcAccessMask = 0;

		renderPassInfo.dependencyCount = 1;
		renderPassInfo.pDependencies = &dependency;


		/*     A better example that illustrates better:
		Let's say that we are implementing shadow mapping and we want to write to a resource (texture)
		and on our main pass we want to read from that texture (to check what is in shadow or don't)


		VkSubpassDependency dep;

		//This will be our shadowmap pass (where we will write the depth to that texture)
		dep.srcSubpass = 0;

		//and this will be our main pass that will consume/read/use that texture made by shadowmap pass.
		//so we need to wait the shadowmap to finish before we can read to the texture it will write
		//thus our dependency (src) is the shadowmap pass and the dependent (dst, us) will be the main pass.
		dep.dstSubpass = 1;
		
		//those are the stages we are asking vulkan to finish on the src subpass (shadowmap pass) BEFORE we can move on to dst subpass (main pass)
		//here we are saying "finish to render and output it to the texture (color attachment)"
		//so only after this texture is done with our contents of the shadowmap that we can proceed to the next subpass (dst/main pass)
		dep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

		 //This lists all passes that WE ARE NOT ALLOWED to execute until the stages in the srcStageMask have completed
		 //This pretty much allows Vulkan to do whatever it wants in this current subpass but when it reaches the FRAGMENT_SHADER_BIT it make it to wait until VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT is done.
		 //In practice it is allowing the main pass (us) to do whatever it wants, for instance, to execute vertex shader and all of sort of stuff
		 //but when the fragment stage finally comes in (it is inside the fragment shader that we will read the shadow map/resource of src) it explicitly tells that it must wait
		 //to the image (texture/shadow map) on the src (shadowmap/dependency) to be written before we can then execute the fragment shader
		dep.dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
		
		//so  the srcStageMask tells what to wait on (finish to write the image)
		//and the dstStageMask tells what will wait  (use/read the image)

		//And here we help the driver by specifying the kind of operations each subpass will perform.
		//So we tell that, as we will be writing to our resource, the src (shadow map) will use a write operation
		dep.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

		//and yet inside our pipeline on GPU, the dst (main pass) will read that resource through a read operation.
		//By telling that our destination WILL read that resource being WRITE, we ensure to flush the result and make it
		//available and visible to our dst subpass. As well as avoiding hazards (by setting barriers)
		dep.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
		*/

		VkCheck(vkCreateRenderPass(m_Device, &renderPassInfo, nullptr, &m_RenderPass));
	}

	void CreateGraphicsPipeline()
	{
	
		std::vector<char> vertShaderCode = ReadFile("res/shaders/vert.spv");
		std::vector<char> fragShaderCode = ReadFile("res/shaders/frag.spv");

		VkShaderModule vertShaderModule = CreateShaderModule(vertShaderCode);
		VkShaderModule fragShaderModule = CreateShaderModule(fragShaderCode);
		
		VkPipelineShaderStageCreateInfo vertShaderStageInfo = {};
		vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		
		//enum explaining in what stage this shader is going to be used for
		//here we say that this code is to be used at the vertex programmable shader stage
		vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;

		//module containing the code
		vertShaderStageInfo.module = vertShaderModule;

		//entry point of this shader. We can use multiple fragment shaders into a single shader module
		//and use different entry points to switch between behaviors.
		//so, for instance, in a single fragment shader code file (module), you can add 3 functions mains
		//and each one do one thing, then you can switch between them just by changing the entrypoint function
		vertShaderStageInfo.pName = "main";

		//This will not be used right now but it is important. You can define constants for the shader
		//in the creation time. For instance, sometimes we have [if]s that depends on constants
		//let's say in case of a debug constant. Then, instead of branching your shader every time (in which is a bad practice)
		//you can pass this constant directly here in the creation time and the compiler will optimize the [if]s that are based on this constant
		//for example, by stripping out the code that is on the else/if counterpart. 
		//This is really useful for plenty of stuff.
		vertShaderStageInfo.pSpecializationInfo = nullptr;

		//Everything here is the same as the vertex basically
		VkPipelineShaderStageCreateInfo fragShaderStageInfo = {};
		fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		fragShaderStageInfo.module = fragShaderModule;
		fragShaderStageInfo.pName = "main";
		fragShaderStageInfo.pSpecializationInfo = nullptr;

		//When actually creating the graphics pipeline, we will use those Shader Stage Create Info
		//to describe what stages do we want to create and how are we going to create them (the code and all details we defined above)
		VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };


		
		//Usually we set all properties beforehand and bake a PSO, but for some stuff you can be flexible and set it in real time
		//this is the case for viewport and scissor (and other stuff) as it is common to be flexible.
		std::vector<VkDynamicState> dynamicStates = {
			VK_DYNAMIC_STATE_VIEWPORT,
			VK_DYNAMIC_STATE_SCISSOR
		};

		VkPipelineDynamicStateCreateInfo dynamicState = {};
		dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
		dynamicState.dynamicStateCount = (uint32_t)dynamicStates.size();
		dynamicState.pDynamicStates = dynamicStates.data();


		//Now we are going to describe the vertex data that we are going to feed to the vertex shader
		//its similar to what we do for vertex attributes on OGL. 
		VkVertexInputBindingDescription bindingDescription = Vertex::GetBindingDescription();
		std::array<VkVertexInputAttributeDescription, 2> attributeDescriptions = Vertex::GetAttributeDescriptions();

		VkPipelineVertexInputStateCreateInfo vertexInputInfo = {};
		vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		vertexInputInfo.vertexBindingDescriptionCount = 1;
		vertexInputInfo.vertexAttributeDescriptionCount = (uint32_t)attributeDescriptions.size();
		vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
		vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

		//Now we are going to describe what kind of shape do we want from our vertex list
		VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
		inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		
		//lets stick with invidiual triangles as usual
		inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		inputAssembly.primitiveRestartEnable = VK_FALSE;

		//Let's setup the viewport, basically its an area where we are going to draw the framebuffer
		//Usually we set from 0, 0 to WIDTH, HEIGHT, indicating that we are just plotting our entire framebuffer onto full screen
		//A good example of split viewport is, for example, when rendering a split screen multiplayer game
		VkViewport viewport = {};
		viewport.x = 0.0f;
		viewport.y = 0.0f;
		viewport.width  = (float)m_RenderTargetExtent.width;
		viewport.height = (float)m_RenderTargetExtent.height;
		//Range of depth for the framebuffer (must be [0,1])
		viewport.minDepth = 0.0f;
		viewport.maxDepth = 1.0f;


		//Now we define a scissor. The viewport defines the place where we will present our framebuffer image
		//so even if we take a small square, it will fit all the image inside this small square
		//in the other size, the Scissor works like a filter where only stuff inside this filter will be shown.
		//so if you have a viewport for the whole screen but a scissor that only covers up half the screen, then
		//the final image will be cropped by half. 
		//The viewport would just squish and stretch the full image to cover only half image, but the scissor will just crop it.
		//Lets define a scissor that covers the whole screen.
		VkRect2D scissor = {};
		scissor.offset = { 0, 0 };
		scissor.extent = m_RenderTargetExtent;

		
		VkPipelineViewportStateCreateInfo viewportState = {};
		viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportState.viewportCount = 1;
		viewportState.scissorCount = 1;
		//Because we already defined that Viewport and Scissor will be dynamic, we don't need to specify it here.
		//If it would not be dynamic, then we would need to specify each member here and make it immutable.
		//it would be something like that:
		///viewportState.pViewports = &viewport;
		///viewportState.pScissors = &scissor;
		//as we don't need, it will be just the count for later setup (we will use the command buffer to set such stuff)


		//Lets setup the rasterizer stage. We will check how it retrieve fragments from within vertex shader shapes
		//and how it performs depth testing, face culling, scissor test etc
		VkPipelineRasterizationStateCreateInfo rasterizer = {};
		rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		
		//This would clamp every fragment that are beyond near and far plane. Useful for shadow mapping. GPU Feature.
		rasterizer.depthClampEnable = VK_FALSE;

		//discard all rasterization. Basically disables any output to framebuffer
		rasterizer.rasterizerDiscardEnable = VK_FALSE;

		//we are going to fill all the geometry with fragments. We could also use it in a wireframe fashion
		//Any mode other than fill requires width and GPU feature.
		rasterizer.polygonMode = VK_POLYGON_MODE_FILL;

		rasterizer.lineWidth = 1.0f;
		
		//backface culling
		rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;

		//define in which order a face is considered to be front
		rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;

		//Depth bias is usually useful for shadow mapping. We are not going to use it right now.
		rasterizer.depthBiasEnable = VK_FALSE;
		rasterizer.depthBiasConstantFactor = 0.0f;
		rasterizer.depthBiasClamp = 0.0f;
		rasterizer.depthBiasSlopeFactor = 0.0f;


		//We are not going to enable multisampling right now (as it requires GPU feature)
		//but this multisample detects whether a pixel was mapped to multiple polygons (on the rasterizer stage)
		//and then blend both fragment shaders to that pixel.
		//This usually occurs on edges (the final of a shape and the background shape, for instance)
		//so this works very well as a anti-aliasing. It is also performant since it doesnt need to render the scene
		//in a higher resolution and then downscale it. It will just invoke additional fragment shader on the borders.
		VkPipelineMultisampleStateCreateInfo multisampling = {};
		multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		multisampling.sampleShadingEnable = VK_FALSE;
		multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
		multisampling.minSampleShading = 1.0f;
		multisampling.pSampleMask = nullptr;
		multisampling.alphaToCoverageEnable = VK_FALSE;
		multisampling.alphaToOneEnable = VK_FALSE;

		//Color blending
		VkPipelineColorBlendAttachmentState colorBlendAttachment = {};
		colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
		colorBlendAttachment.blendEnable = VK_TRUE;
		colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
		colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
		colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
		colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
		colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
		colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;

		VkPipelineColorBlendStateCreateInfo colorBlending = {};
		colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		colorBlending.logicOpEnable = VK_FALSE;
		colorBlending.logicOp = VK_LOGIC_OP_COPY; // Optional
		colorBlending.attachmentCount = 1;
		colorBlending.pAttachments = &colorBlendAttachment;
		colorBlending.blendConstants[0] = 0.0f; // Optional
		colorBlending.blendConstants[1] = 0.0f; // Optional
		colorBlending.blendConstants[2] = 0.0f; // Optional
		colorBlending.blendConstants[3] = 0.0f; // Optional

		//This is used to pass Uniforms to shaders. We will create an empty one for now.
		VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
		pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pipelineLayoutInfo.setLayoutCount = 0;
		pipelineLayoutInfo.pSetLayouts = nullptr;
		pipelineLayoutInfo.pushConstantRangeCount = 0;
		pipelineLayoutInfo.pPushConstantRanges = nullptr;

		VkCheck(vkCreatePipelineLayout(m_Device, &pipelineLayoutInfo, nullptr, &m_PipelineLayout));


		VkGraphicsPipelineCreateInfo pipelineInfo = {};
		pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		pipelineInfo.stageCount = 2;
		pipelineInfo.pStages = shaderStages;

		pipelineInfo.pVertexInputState = &vertexInputInfo;
		pipelineInfo.pInputAssemblyState = &inputAssembly;
		pipelineInfo.pViewportState = &viewportState;
		pipelineInfo.pRasterizationState = &rasterizer;
		pipelineInfo.pMultisampleState = &multisampling;
		pipelineInfo.pDepthStencilState = nullptr;
		pipelineInfo.pColorBlendState = &colorBlending;
		pipelineInfo.pDynamicState = &dynamicState;

		pipelineInfo.layout = m_PipelineLayout;
		
		pipelineInfo.renderPass = m_RenderPass;
		pipelineInfo.subpass = 0;

		//We can specify that we are creating an pipeline derived from another pipeline.
		//when this happens, creation time and time between swapping derivate pipelines are faster.
		//This usually is made when a pipeline have a lot in common with another pipeline.
		//VK_PIPELINE_CREATE_DERIVATIVE_BIT must be set on VkGraphicsPipelineCreateInfo.
		pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
		pipelineInfo.basePipelineIndex = -1;

		//We are not specifying a Pipeline Cache right now but we will in the future. We use this to cache pipelines and speed up creation time
		VkCheck(vkCreateGraphicsPipelines(m_Device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_GraphicsPipeline));

		//destruction of what is not useful anymore
		vkDestroyShaderModule(m_Device, vertShaderModule, nullptr);
		vkDestroyShaderModule(m_Device, fragShaderModule, nullptr);
	}

	VkShaderModule CreateShaderModule(const std::vector<char>& byteCode)
	{
		VkShaderModuleCreateInfo moduleCreateInfo = {};
		moduleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
		moduleCreateInfo.codeSize = byteCode.size();
		moduleCreateInfo.pCode = reinterpret_cast<const uint32_t*>(byteCode.data());
		
		VkShaderModule shaderModule;

		VkCheck(vkCreateShaderModule(m_Device, &moduleCreateInfo, nullptr, &shaderModule));

		return shaderModule;
	}

	void CreateSwapChainImageViews()
	{
		m_RenderTargetViews.resize(m_RenderTargets.size());

		for (uint32_t i = 0; i < m_RenderTargets.size(); i++)
		{
			VkImageViewCreateInfo viewCreateInfo = {};
			viewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
			viewCreateInfo.image = m_RenderTargets[i];

			//it can be 1D, 2D, 3D and cube maps
			viewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;

			viewCreateInfo.format = m_RenderTargetFormat;

			viewCreateInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
			viewCreateInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
			viewCreateInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
			viewCreateInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;

			//we will define an image to be used as color, but without any mips or multiple layers
			//so we set the base mip as 0 and we say that we only have 1 (the base)
			//the same happens for the layer
			//Layer is usually used for VR and such applications
			viewCreateInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			viewCreateInfo.subresourceRange.baseMipLevel = 0;
			viewCreateInfo.subresourceRange.levelCount = 1;
			viewCreateInfo.subresourceRange.baseArrayLayer = 0;
			viewCreateInfo.subresourceRange.layerCount = 1;

			VkCheck(vkCreateImageView(m_Device, &viewCreateInfo, nullptr, &m_RenderTargetViews[i]));
		}
	}

	void CreateSwapChain()
	{
		SwapChainSupportDetails swapChainSupportDetails = QuerySwapChainSupport(m_PhysicalGPU);

		VkSurfaceFormatKHR surfaceFormatSupported = ChooseSwapSurfaceFormat(swapChainSupportDetails.formats);
		VkPresentModeKHR presentModeSupported = ChooseSwapPresentMode(swapChainSupportDetails.presentModes);
		VkExtent2D extendSupported = ChooseSwapExtent(swapChainSupportDetails.capabilities);

		//set number of framebuffers for the minimum supported (usually 2)
		//It is also good adviced to use minimum + 1 because sometimes we need finish some internal operations
		//before replacing the current backbuffer and begin to draw to it. 
		uint32_t imageCount = swapChainSupportDetails.capabilities.minImageCount + 1;

		//When maxImageCount is zero, it means that we dont have a max.
		//In this case, if we have a max, we will then remember to clamp to it, so we can't never have more than the maximum
		if (swapChainSupportDetails.capabilities.maxImageCount > 0 && imageCount > swapChainSupportDetails.capabilities.maxImageCount)
		{
			imageCount = swapChainSupportDetails.capabilities.maxImageCount;
		}

		VkSwapchainCreateInfoKHR swapChainCreateInfo = {};
		swapChainCreateInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
		swapChainCreateInfo.surface = m_WindowSurface;

		swapChainCreateInfo.minImageCount = imageCount;
		swapChainCreateInfo.imageFormat = surfaceFormatSupported.format;
		swapChainCreateInfo.imageColorSpace = surfaceFormatSupported.colorSpace;
		swapChainCreateInfo.imageExtent = extendSupported;

		//The imageArrayLayers specifies the amount of layers each image consists of. 
		//This is always 1 unless you are developing a stereoscopic 3D application (such as VR or that blue/red 3D)
		swapChainCreateInfo.imageArrayLayers = 1;

		//We will say that we will just write to color, just a regular image write.
		//In the future we may want to only write to a depth buffer, for instance
		//Then further steps are necessary, such as using it as VK_IMAGE_USAGE_TRANSFER_DST_BIT so we can use that image later
		swapChainCreateInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

		//Let's assume the graphic queue is the same as the presentation queue. With this, we will set the share mode as Exclusive.
		//Exclusive means that a queue owns an image and have to explicitly transfer the ownership to other queue.
		//So we would have to explicitly transfer the ownership of the image from the graphic queue to the presentation queue
		//in case they are not the same. This works like an memory/resource barrier.
		//Alternatively we have VK_SHARING_MODE_CONCURRENT, that handles this internally for us. The performance is worse though.
		//Also, Concurrent asks the user to specify what families the ownership will be shared between. So it needs an array of family indices (types)
		//In this case, we are assuming that the graphic queue is the same as the presentation queue as it is common for almost
		//every PC that supports graphics. So we will not lose too much sleep hours here. 
		//Ownership (resource/memory barriers) and stuff will be presented later.
		swapChainCreateInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;

		//Specify a transform (if supported), such as 90 degree rotation, flip etc. Before presenting to screen.
		//If you don't want any of that, just use the currentTransform as the value.
		swapChainCreateInfo.preTransform = swapChainSupportDetails.capabilities.currentTransform;

		//We can decide if we want the alpha channel to be used to blend with other windows
		//but we will ignore this and focus on our own game
		swapChainCreateInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;

		swapChainCreateInfo.presentMode = presentModeSupported;

		//This mean that we don't care about occluded pixels. This can happen if another window
		//is in front of our pixels or the window is cropped by the screen.
		//we possibly get more performance by enabling clipping.
		swapChainCreateInfo.clipped = VK_TRUE;

		//When recreating a swap chain (in case it gets invalid by resizing for instance)
		//we must also specify the older swap chain. This will be discussed later
		//but for now lets assume we will ever have only one swap chain.
		swapChainCreateInfo.oldSwapchain = VK_NULL_HANDLE;

		VkCheck(vkCreateSwapchainKHR(m_Device, &swapChainCreateInfo, nullptr, &m_SwapChain));
	
		vkGetSwapchainImagesKHR(m_Device, m_SwapChain, &imageCount, nullptr);
		m_RenderTargets.resize(imageCount);
		vkGetSwapchainImagesKHR(m_Device, m_SwapChain, &imageCount, m_RenderTargets.data());;

		m_RenderTargetFormat = surfaceFormatSupported.format;
		m_RenderTargetExtent = extendSupported;
	}

	void CleanupSwapChain()
	{
		for (int i = 0; i < m_SwapChainFramebuffers.size(); i++)
			vkDestroyFramebuffer(m_Device, m_SwapChainFramebuffers[i], nullptr);

		for (uint32_t i = 0; i < m_RenderTargetViews.size(); i++)
			vkDestroyImageView(m_Device, m_RenderTargetViews[i], nullptr);

		vkDestroySwapchainKHR(m_Device, m_SwapChain, nullptr);
	}

	void RecreateSwapChain()
	{
		int width = 0, height = 0;
		glfwGetFramebufferSize(m_Window, &width, &height);

		while (width == 0 || height == 0)
		{
			glfwGetFramebufferSize(m_Window, &width, &height);

			if (glfwWindowShouldClose(m_Window))
				return;

			//instead of peeking events and keeping to update the game (in this case, the , we will sleep the thread until there is another event
			//this event would be a Framebuffer Resize from GLFW, probably maximizing it, so we can also back to update our loop
			//if we want to, for instance, pause the rendering but keep with physics loop or something like that, we would need an alternative for this part that doesnt hang the program
			glfwWaitEvents();
		}


		vkDeviceWaitIdle(m_Device);

		CleanupSwapChain();

		CreateSwapChain();
		CreateSwapChainImageViews();
		CreateFramebuffers();
	}

	VkSurfaceFormatKHR ChooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats)
	{
		for (const VkSurfaceFormatKHR& availableFormat : availableFormats)
		{
			if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
			{
				return availableFormat;
			}
		}

		return availableFormats[0];
	}

	VkPresentModeKHR ChooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes)
	{
		for (const VkPresentModeKHR& availablePresent : availablePresentModes)
		{
			if (availablePresent == VK_PRESENT_MODE_MAILBOX_KHR)
			{
				return availablePresent;
			}
		}

		//Fifo always guaranteed to be available.
		return VK_PRESENT_MODE_FIFO_KHR;
	}

	VkExtent2D ChooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities)
	{
		if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max())
			return capabilities.currentExtent;
		
		int width, height;

		glfwGetFramebufferSize(m_Window, &width, &height);

		VkExtent2D actualExtent = { uint32_t(width), (uint32_t(height)) };

		actualExtent.width = clamp<uint32_t>(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
		actualExtent.height = clamp<uint32_t>(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

		return actualExtent;
	}

	SwapChainSupportDetails QuerySwapChainSupport(VkPhysicalDevice device)
	{
		SwapChainSupportDetails details;
		vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, m_WindowSurface, &details.capabilities);

		uint32_t formatCount = 0;
		vkGetPhysicalDeviceSurfaceFormatsKHR(device, m_WindowSurface, &formatCount, nullptr);

		if (formatCount != 0)
		{
			details.formats.resize(formatCount);
			vkGetPhysicalDeviceSurfaceFormatsKHR(device, m_WindowSurface, &formatCount, details.formats.data());
		}

		uint32_t presentModeCount = 0;
		vkGetPhysicalDeviceSurfacePresentModesKHR(device, m_WindowSurface, &presentModeCount, nullptr);

		if (presentModeCount != 0)
		{
			details.presentModes.resize(presentModeCount);
			vkGetPhysicalDeviceSurfacePresentModesKHR(device, m_WindowSurface, &presentModeCount, details.presentModes.data());
		}


		return details;
	}

	void CreateWindowSurface()
	{
		
		//We will use the agnostic part of GLFW and ask it to create a Surface for us.
		//but if we were on Windows, we could create like so:
		//VkWin32SurfaceCreateInfoKHR createInfo{};
		//createInfo.sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
		//createInfo.hwnd = glfwGetWin32Window(window);
		//createInfo.hinstance = GetModuleHandle(nullptr);
		//vkCreateWin32SurfaceKHR(instance, &createInfo, nullptr, &surface)
		//for linux it would be the same but replacing vkCreateWin32SurfaceKHR for vkCreateXcbSurfaceKHR 

		VkCheck(glfwCreateWindowSurface(m_Instance, m_Window, nullptr, &m_WindowSurface));
	}

	void CreateGPULogicalHandle()
	{
		VkDeviceQueueCreateInfo queueCreateInfo = {};
		queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
		queueCreateInfo.queueFamilyIndex = m_GraphicsQueueFamilyType;
		queueCreateInfo.queueCount = 1;

		//set priority for the scheduling of command buffer execution in case we have multiple queues

		float queuePriority = 1.0f;
		queueCreateInfo.pQueuePriorities = &queuePriority;

		VkPhysicalDeviceFeatures deviceFeatures = {};

		VkDeviceCreateInfo deviceCreateInfo = {};
		deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
		deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
		deviceCreateInfo.queueCreateInfoCount = 1;

		deviceCreateInfo.pEnabledFeatures = &deviceFeatures;

#ifdef VKDEBUG
		deviceCreateInfo.enabledLayerCount = 1;
		deviceCreateInfo.ppEnabledLayerNames = &g_RequiredValidationLayer;
#else
		deviceCreateInfo.enabledLayerCount = 0;
#endif

		//For getting swap chain ext
		std::vector<const char*> deviceRequiredExtensions = GetDeviceRequiredExtensions();

		uint32_t extensionCount = 0;
		vkEnumerateDeviceExtensionProperties(m_PhysicalGPU, nullptr, &extensionCount, nullptr);

		std::vector<VkExtensionProperties> availableDeviceExtensions(extensionCount);
		vkEnumerateDeviceExtensionProperties(m_PhysicalGPU, nullptr, &extensionCount, availableDeviceExtensions.data());

		for (int x = 0; x < deviceRequiredExtensions.size(); x++)
		{
			bool found = false;

			for (int i = 0; i < availableDeviceExtensions.size(); i++)
			{
				std::string glfwRequiredExtension(deviceRequiredExtensions[x]);
				if (glfwRequiredExtension == availableDeviceExtensions[i].extensionName)
				{
					found = true;
					break;
				}
			}

			if (!found)
			{
				std::cerr << "Required extension not found/supported! Extension: " << deviceRequiredExtensions[x] << std::endl;
				__debugbreak();
			}
		}

		//Now that we support swap chain, lets check if it has at least 1 image format and presentation mode
		//it is important to only query for swap chain details after checking if we support the extension
		SwapChainSupportDetails swapChainSupport = QuerySwapChainSupport(m_PhysicalGPU);
		if (swapChainSupport.formats.empty() || swapChainSupport.presentModes.empty())
		{
			std::cerr << "Swap Chain supported but not suitable. " << std::endl;
			__debugbreak();
		}

		//So lets create the device with swap chain extension
		//we will be creating an swap chain with this device later. 
		//and it will be automatically attached to it and the corresponding window surface.
		//so it will now from which device to which window to draw.
		deviceCreateInfo.enabledExtensionCount = (uint32_t)deviceRequiredExtensions.size();
		deviceCreateInfo.ppEnabledExtensionNames = deviceRequiredExtensions.data();

		//Create our logical handle to our gpu.
		VkCheck(vkCreateDevice(m_PhysicalGPU, &deviceCreateInfo, nullptr, &m_Device));

		//the 0 is because we can create more than one queue of the same type. In our case we only create one.
		//we just need to pass the type of the queue we created, so Vulkan knows which one to return
		//we already queried earlier to see if we support a family type with graphics capabilities and got this type.
		vkGetDeviceQueue(m_Device, m_GraphicsQueueFamilyType, 0, &m_GraphicsQueueHandle);
	}

	void PickPhysicalDevice()
	{
		uint32_t deviceCount = 0;
		vkEnumeratePhysicalDevices(m_Instance, &deviceCount, nullptr);

		if (!deviceCount)
		{
			std::cerr << "No GPUs compatible with Vulkan found!" << std::endl;
			__debugbreak();
		}

		std::vector<VkPhysicalDevice> devices(deviceCount);
		vkEnumeratePhysicalDevices(m_Instance, &deviceCount, devices.data());

		std::map<int, VkPhysicalDevice> candidatesAndScore;

		//Device name, type and supported vulkan version
		for (const VkPhysicalDevice& device : devices)
		{
			VkPhysicalDeviceProperties deviceProperties;
			vkGetPhysicalDeviceProperties(device, &deviceProperties);

			//texture compression, 64 bit floats, multi viewport rendering (VR) etc..
			VkPhysicalDeviceFeatures deviceFeatures;
			vkGetPhysicalDeviceFeatures(device, &deviceFeatures);

			int score = 0;
			
			//+1000 if it is offboard
			if (deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
				score += 1000;

			//if we have more than on offboard, let's stick up with the better. Max 2D texture size is a good measurement 
			score += deviceProperties.limits.maxImageDimension2D;

			//If we don't have Geometry Shaders as a feature on this GPU, let's zero the score.
			score *= deviceFeatures.geometryShader;

			//Let's check if our GPU has a Queue Family in which support graphics commands (such as draw)
			//as this is a must, we will zero the score if it doesn't support.
			uint32_t queueFamilyCount = 0;
			vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);
			std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
			vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());
			
			uint32_t graphicsFamily = (uint32_t)-1;

			for (uint32_t i = 0; i < queueFamilies.size(); i++)
			{
				if (queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
				{
					//We don't have any sort of Handle for each queue
					//The index of the Family array returned by Vulkan IS the identifier of the family.
					//So if the Graphics family is Index 0 on the array, the internal identifier IS 0
					//So when creating a Logical Device, passing 0 as the family, will pass the actual Graphics family.
					//Vulkan has the queue family as a identifier, but when creating the logical device, the family queue
					//will actually be created for us. And then we use the ID to get the handle of the internal created queue.
					//So for instance, if we pass the defined queue 0 (that we saw we support), it will retrieve where it created
					//the family type #0 for us. i.e: the actual handle of the graphics queue family.
					//these types definitions are created beforehand by vulkan for each gpu. So we just need to check
					//if we have any type of family that supports graphics, and then we instantiate a family of that type.
					
					//Lets check if this queue supports presentation
					//usually, when the queue supports graphics, it does support presentation
					//but the inverse is not true. So we have queues that support presentation but does not support graphics.
					//so now that we have a graphics queue, lets just ensure that it does support presentation:
					VkBool32 presentSupport = false;
					vkGetPhysicalDeviceSurfaceSupportKHR(device, i, m_WindowSurface, &presentSupport);

					if (!presentSupport)
						continue;


					graphicsFamily = i;
					break;
				}
			}

			if (graphicsFamily == uint32_t(-1))
			{
				score *= 0;
			}
			else
			{
				m_GraphicsQueueFamilyType = graphicsFamily;
			}

			candidatesAndScore[score] = device;
		
		}

		if (candidatesAndScore.rbegin()->first > 0)
		{
			m_PhysicalGPU = candidatesAndScore.rbegin()->second;

			VkPhysicalDeviceProperties deviceProperties;
			vkGetPhysicalDeviceProperties(m_PhysicalGPU, &deviceProperties);

			std::cout << "Selected GPU: " << deviceProperties.deviceName << std::endl;
		}
		else
		{
			std::cerr << "No suitable GPU found!" << std::endl;
			__debugbreak();
		}

	}

	std::vector <const char*> GetDeviceRequiredExtensions() const
	{
		std::vector<const char*> requiredExtensions;
		//Add swap chain ext
		requiredExtensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);

		return requiredExtensions;
	}

	std::vector<const char*> GetInstanceRequiredExtensions() const
	{
		//Get GLFW required extensions
		const char** glfwExtensions;
		uint32_t glfwExtensionCount = 0;
		glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

		std::vector<const char*> requiredExtensions(glfwExtensions, glfwExtensions + glfwExtensionCount);


#ifdef VKDEBUG
		//if we are on debug builds, add "VK_EXT_debug_utils" to the list of required extensions, so we can load up validation layers
		requiredExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
#endif

		return requiredExtensions;
	}

#ifdef VKDEBUG
	void PopulateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo)
	{
		createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;

		createInfo.messageSeverity = /* VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | */
			VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
			VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;

		createInfo.messageType = /* VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | */
			VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
			VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;

		createInfo.pfnUserCallback = VulkanValidationLayerLogCallback;

		//This lets you set any data you want. This data will be passed to VulkanValidationLayerLogCallback through pUserData parameter.
		//we could set this as HelloTriangleApp for instance.
		createInfo.pUserData = nullptr;
	}

	void SetupDebugMessenger()
	{
		VkDebugUtilsMessengerCreateInfoEXT createInfo = {};
		PopulateDebugMessengerCreateInfo(createInfo);

		VkCheck(CreateDebugUtilsMessengerEXT(m_Instance, &createInfo, nullptr, &m_DebugMessenger));
	}
#endif

	void InitWindow()
	{
		glfwInit();

		//Explicit tell GLFW to not create OGL context
		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

		glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

		m_Window = glfwCreateWindow(APP_WIDTH, APP_HEIGHT, "Vulkan!", nullptr, nullptr);

		glfwSetWindowUserPointer(m_Window, this);
		glfwSetFramebufferSizeCallback(m_Window, FrameBufferResizeCallback);
	}

	static void FrameBufferResizeCallback(GLFWwindow* window, int width, int height)
	{
		HelloTriangleApp* app = reinterpret_cast<HelloTriangleApp*>(glfwGetWindowUserPointer(window));
		
		app->m_WindowResized = true;
	}

	void InitVulkan()
	{
		// ------------------------------------------------------------------------ Create instance
		VkApplicationInfo appInfo = {};
		appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
		appInfo.pApplicationName = "Hello Triangle";
		appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
		appInfo.pEngineName = "No Engine";
		appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
		appInfo.apiVersion = VK_API_VERSION_1_0;

		VkInstanceCreateInfo createInfo = {};
		createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
		createInfo.pApplicationInfo = &appInfo;
		
		//Retrieve required extensions
		std::vector<const char*> requiredExtensions = GetInstanceRequiredExtensions();
		

		//Get all supported extensions 
		uint32_t supportedExtensionCount = 0;

		vkEnumerateInstanceExtensionProperties(nullptr, &supportedExtensionCount, nullptr);

		std::vector<VkExtensionProperties> vkSupportedExtensions(supportedExtensionCount);

		vkEnumerateInstanceExtensionProperties(nullptr, &supportedExtensionCount, vkSupportedExtensions.data());

		//Check if we support all required extensions
		for (int x = 0; x < requiredExtensions.size(); x++)
		{
			bool found = false;

			for (int i = 0; i < vkSupportedExtensions.size(); i++)
			{
				std::string glfwRequiredExtension(requiredExtensions[x]);
				if (glfwRequiredExtension == vkSupportedExtensions[i].extensionName)
				{
					found = true;
					break;
				}
			}

			if (!found)
			{
				std::cerr << "Required extension not found/supported! Extension: " << requiredExtensions[x] << std::endl;
				__debugbreak();
			}

		}

		//If so, create the instance
		createInfo.enabledExtensionCount = (uint32_t)requiredExtensions.size();
		createInfo.ppEnabledExtensionNames = requiredExtensions.data();


#ifdef VKDEBUG

		//Check for validation layers if in _DEBUG build
		uint32_t layerCount = 0;
		vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
		std::vector<VkLayerProperties> availableLayers(layerCount);
		vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());


		bool foundValidationLayer = false;
		//Quick layer check using strings, no worries on perf since this will not happen on real time.
		for (int i = 0; i < availableLayers.size(); i++)
		{
			if (availableLayers[i].layerName == std::string(g_RequiredValidationLayer))
			{
				foundValidationLayer = true;
				break;
			}
		}

		if (!foundValidationLayer)
		{
			std::cerr << "Validation/debug layers were requested but none are available!";
			__debugbreak();
		}

		//I will not handle it on a separated callback for now, let it flush to the std output.
		createInfo.enabledLayerCount = 1;
		createInfo.ppEnabledLayerNames = &g_RequiredValidationLayer;

		//This time, Vulkan will create an internal DebugMessenger just to debug Create/Destroy call to the Instance
		//For this, we only need to pass the CreateInfo so it will know where and what to display to us.
		//After the Instance destruction, this will be automatically cleaned up for us.
		//PS: Our own debug messenger, we still have to explicitly call "CreateDebugUtilsMessengerEXT" and pass the Instance as arg.
		VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo = {};
		PopulateDebugMessengerCreateInfo(debugCreateInfo);
		
		createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*)&debugCreateInfo;

#else
		createInfo.enabledLayerCount = 0;
#endif

		VkCheck(vkCreateInstance(&createInfo, nullptr, &m_Instance));

		// ------------------------------------------------------------------------ Create instance
	}

	void Update()
	{
		while (!glfwWindowShouldClose(m_Window))
		{
			glfwPollEvents();
			Draw();
		}

		vkDeviceWaitIdle(m_Device);
	}

	void Draw()
	{
		//We need to wait on the currently frame to finish. When the Queue is done, it will signal this fence 
		//and we can effectively unblock the CPU. Until there, we will block this thread
		vkWaitForFences(m_Device, 1, &m_InFlightFrameFences[s_CurrentFrame], VK_TRUE, UINT64_MAX);

		uint32_t imageIndex;
		//we will ask for a new image from the swap chain. When this image is retrieved, this Semaphore will be signaled.
		VkResult result = vkAcquireNextImageKHR(m_Device, m_SwapChain, UINT64_MAX, m_RenderTargetAvailableSemaphores[s_CurrentFrame], VK_NULL_HANDLE, &imageIndex);

		//We only fall here if we simply can't draw to the new size at all.
		//Usually resizes doesn't leads us here
		if (result == VK_ERROR_OUT_OF_DATE_KHR)
		{
			RecreateSwapChain();
			return;
		}
		
		// Only reset the fence if we are submitting work
		vkResetFences(m_Device, 1, &m_InFlightFrameFences[s_CurrentFrame]);

		vkResetCommandBuffer(m_CommandBuffers[s_CurrentFrame], 0);

		//We are going to record the commands, we are not going to execute them right now. So even though we don't have a render target right now (semaphore is not signaled)
		//we can still record the commands referencing that render target.
		//When everything is done and we have that render target, the command list will be submitted (as we fill that semaphore on the VkSubmitInfo structure)
		RecordCommands(m_CommandBuffers[s_CurrentFrame], imageIndex);

		VkSubmitInfo submitInfo = {};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

		VkSemaphore waitSemaphores[] = { m_RenderTargetAvailableSemaphores[s_CurrentFrame] };
		VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
		
		submitInfo.waitSemaphoreCount = 1;

		//Specify which semaphore to wait on before execution
		submitInfo.pWaitSemaphores = waitSemaphores;

		//Each index of waitStages[] corresponds to the index of waitSemaphores[]
		//this is, we can have different semaphores to make different stages of the pipeline to wait.
		//In our case, we are waiting to have a new image, to output the color onto (i.e: to write to)
		//Theoretically, we are only waiting an image to proceed the write step, so we should be executing the vertex shader and such even without an image to write.
		submitInfo.pWaitDstStageMask = waitStages;

		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &m_CommandBuffers[s_CurrentFrame];
		
		
		//In a submit batch, you send multiple command lists, for each command list we have a corresponding semaphore
		//each index on the command list array, is an index com the semaphore array
		//that way we can also keep track when a specific command list finishes it work
		//in our case we only have one command list, so we will only use one semaphore
		VkSemaphore signalSemaphores[] = { m_RenderFinishedSemaphores[s_CurrentFrame] };
		submitInfo.signalSemaphoreCount = 1;

		//We have Semaphores to keep sync between objects on the GPU and a Fence to keep sync between GPU and CPU
		//So for instance, the Presentinfo on SwapChain are waiting the frame to be drawn (aka the command list to finish)
		//so it can present to the screen. So for that we will use a Semaphore to warn the Swapchain that we finished.
		//This is more efficient because almost everything of this are happening inside the GPU, so we don't need to signal back to the CPU
		//and this is not exclusively just for when finishing all the work on Queue but for a command list.
		submitInfo.pSignalSemaphores = signalSemaphores;

		//In the other hand, the Queue provide us a way to warn the CPU when all the work is done, this is, the frame (and all command lists).
		//It then signals this Fence telling that the frame is over.
		//Then as the CPU is waiting on this Fence, it knows that it can begin the process to write again to this command list
		//as it is not being used anymore.
		//If we would not wait on this fence, we would write to a command list that it is still being executed thus overriding the commands
		VkCheck(vkQueueSubmit(m_GraphicsQueueHandle, 1, &submitInfo, m_InFlightFrameFences[s_CurrentFrame]));

		VkPresentInfoKHR presentInfo = {};
		presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

		presentInfo.waitSemaphoreCount = 1;
		//We tell the present info that we wait to wait on this semaphore before presenting
		//this semaphore is the one we signal when we finish to write to our render target.
		presentInfo.pWaitSemaphores = signalSemaphores;

		VkSwapchainKHR swapChains[] = { m_SwapChain };
		
		presentInfo.swapchainCount = 1;
		presentInfo.pSwapchains = swapChains;
		presentInfo.pImageIndices = &imageIndex;
		presentInfo.pResults = nullptr;

		result = vkQueuePresentKHR(m_GraphicsQueueHandle, &presentInfo);

		//Ensure this is after QueuePresent because we need the fences to be signaled on the next draw
		//otherwise it will deadlock on the wait.
		//For all types of resize, create a new swapchain with the new size.
		//next time, the commands will be recorded to the new images just as fine
		if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || m_WindowResized)
		{
			m_WindowResized = false;
			RecreateSwapChain();
		}


		//Now that we have more than one command buffer, we can properly use multiple images from swap chain
		//see, previously we had to wait for the image to render and present, meanwhile we would block the CPU until the Queue (GPU) finished everything. Then after that we would wait a little more to get an image from the swap chain.
		//Now, we can record and send the image to render and meanwhile, we are not waiting in the same fence anymore. So we can take another image, record it and submit to the Queue (GPU). We would then wait for the first image to become available again
		//while we wait, the GPU is finishing to present it and going to render the next frame and present it, so we can write again to the first image.
		//Now we effectively uses double buffering, when one image is being rendered, we are recording another. Last time we had to wait till the image were rendered and presented to then idle the GPU and record new commands. 
		//this time the GPU always have a command buffer to execute while returning the previous image back to CPU.
		//this is still single threaded but using a different render target for CPU and GPU. (just like we do on dx12)
		s_CurrentFrame = (s_CurrentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
	}

	void Destroy()
	{
		CleanupSwapChain();

		vkDestroyBuffer(m_Device, m_VertexBuffer, nullptr);
		vkFreeMemory(m_Device, m_VertexBufferMemory, nullptr);

		for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
		{
			vkDestroySemaphore(m_Device, m_RenderFinishedSemaphores[i], nullptr);
			vkDestroySemaphore(m_Device, m_RenderTargetAvailableSemaphores[i], nullptr);
			vkDestroyFence(m_Device, m_InFlightFrameFences[i], nullptr);
		}

		vkDestroyCommandPool(m_Device, m_CommandPool, nullptr);
		vkDestroyCommandPool(m_Device, m_TemporaryCommandPool, nullptr);

		vkDestroyPipeline(m_Device, m_GraphicsPipeline, nullptr);
		vkDestroyPipelineLayout(m_Device, m_PipelineLayout, nullptr);
		vkDestroyRenderPass(m_Device, m_RenderPass, nullptr);

		vkDestroyDevice(m_Device, nullptr);

#ifdef VKDEBUG
		DestroyDebugUtilsMessengerEXT(m_Instance, m_DebugMessenger, nullptr);
#endif
		vkDestroySurfaceKHR(m_Instance, m_WindowSurface, nullptr);
		vkDestroyInstance(m_Instance, nullptr);

		glfwDestroyWindow(m_Window);
		glfwTerminate();
	}

	static VKAPI_ATTR VkBool32 VKAPI_CALL VulkanValidationLayerLogCallback(VkDebugUtilsMessageSeverityFlagBitsEXT  messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData)
	{
		std::cerr << "----- VULKAN VALIDATION LAYER: " << pCallbackData->pMessage << std::endl;

		return VK_FALSE;
	}

private:
	GLFWwindow* m_Window;

	VkSurfaceKHR m_WindowSurface;

#ifdef VKDEBUG
	VkDebugUtilsMessengerEXT m_DebugMessenger;
#endif

	VkInstance m_Instance;
	
	VkPhysicalDevice m_PhysicalGPU = VK_NULL_HANDLE;
	
	uint32_t m_GraphicsQueueFamilyType = uint32_t(-1);
	VkDevice m_Device;
	VkQueue m_GraphicsQueueHandle;

	VkSwapchainKHR m_SwapChain;
	std::vector<VkImage> m_RenderTargets;
	std::vector<VkImageView> m_RenderTargetViews;
	std::vector<VkFramebuffer> m_SwapChainFramebuffers;

	VkFormat m_RenderTargetFormat;
	VkExtent2D m_RenderTargetExtent;

	VkRenderPass m_RenderPass;
	VkPipelineLayout m_PipelineLayout;

	//Manages buffer memories and also creates command buffers.
	VkCommandPool m_CommandPool, m_TemporaryCommandPool;
	std::vector<VkCommandBuffer> m_CommandBuffers;

	VkPipeline m_GraphicsPipeline;

	//sync objects
	std::vector<VkSemaphore> m_RenderTargetAvailableSemaphores;
	std::vector<VkSemaphore> m_RenderFinishedSemaphores;
	std::vector<VkFence> m_InFlightFrameFences;

	VkBuffer m_VertexBuffer;
	VkDeviceMemory m_VertexBufferMemory;

	bool m_WindowResized = false;
};


int main()
{
	HelloTriangleApp app;
	app.Run();

	return 0;
}