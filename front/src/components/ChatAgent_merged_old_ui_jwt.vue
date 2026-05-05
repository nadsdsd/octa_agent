<template>
  <div class="flex h-[90vh] max-w-7xl mx-auto bg-white rounded-xl shadow-2xl overflow-hidden border border-gray-200">
    
    <aside class="w-64 bg-gray-50 border-r border-gray-200 flex flex-col flex-shrink-0">
      <div class="p-5 border-b border-gray-200 flex items-center justify-between bg-white">
        <div class="flex items-center">
          <div class="w-10 h-10 rounded-full bg-indigo-100 text-indigo-700 flex items-center justify-center font-bold text-lg mr-3 shadow-sm">
            {{ userInitial }}
          </div>
          <div class="overflow-hidden">
            <p class="font-bold text-gray-800 truncate" :title="authState.username">{{ authState.username || '未登录' }}</p>
            <p class="text-xs text-gray-500 capitalize">{{ authState.role === 'doctor' ? '医生' : '患者' }}</p>
          </div>
        </div>
        <button @click="handleLogout" class="text-gray-400 hover:text-red-500 hover:bg-red-50 p-2 rounded-lg transition-colors" title="退出登录">
          <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1"></path></svg>
        </button>
      </div>

      <div class="flex-1 overflow-y-auto p-3">
        <div class="flex items-center justify-between mb-3 px-2 mt-2">
          <h3 class="text-xs font-semibold text-gray-400 uppercase tracking-wider">会话历史</h3>
          <button class="text-indigo-500 hover:bg-indigo-50 p-1 rounded transition-colors" title="新建会话">
            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v16m8-8H4"></path></svg>
          </button>
        </div>
        <div class="space-y-1">
          <div class="p-3 bg-indigo-50 text-indigo-700 border border-indigo-100 rounded-lg text-sm font-medium cursor-pointer shadow-sm flex items-center">
            <svg class="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"></path></svg>
            <span class="truncate">当前诊断: {{ currentSessionId }}</span>
          </div>
        </div>
      </div>
    </aside>

    <div class="flex-1 flex flex-col min-w-0 bg-gray-50">
      <header class="bg-white px-6 py-4 border-b border-gray-200 flex items-center shadow-sm z-10">
        <div class="w-10 h-10 bg-indigo-600 rounded-full flex items-center justify-center text-white font-bold text-xl mr-4 shadow-md">AI</div>
        <div>
          <h1 class="text-xl font-bold text-gray-800">多智能体协同会诊中枢</h1>
          <p class="text-xs text-green-500 font-medium flex items-center">
            <span class="w-2 h-2 rounded-full bg-green-500 mr-1 animate-pulse"></span> SSE 打字机流开启
          </p>
        </div>
      </header>

      <main class="flex-1 overflow-y-auto p-6 space-y-8" ref="chatContainer">
        <div v-for="(msg, index) in messages" :key="index" class="flex flex-col">
          
          <div v-if="msg.role === 'user'" class="self-end max-w-[80%]">
            <div class="bg-indigo-600 text-white rounded-2xl rounded-tr-sm px-5 py-3 shadow-md">
              <img v-if="msg.image" :src="msg.image" class="w-32 h-32 object-cover rounded-md mb-2 border border-indigo-400" />
              <p>{{ msg.text }}</p>
            </div>
          </div>

          <div v-else class="self-start w-full">
            <div class="flex items-start">
              <div class="w-10 h-10 rounded-full bg-white border border-gray-200 shadow-sm flex items-center justify-center mt-1 mr-4 flex-shrink-0 text-xl">🤖</div>
              <div class="flex-1">
                <div class="relative border-l-2 border-indigo-200 ml-4 pl-8 py-2 space-y-8">
                  
                  <div v-if="msg.statusMsg" class="relative">
                    <span class="absolute -left-[41px] flex h-5 w-5 items-center justify-center rounded-full bg-blue-100 ring-4 ring-gray-50">
                      <span class="h-2.5 w-2.5 rounded-full bg-blue-500 animate-ping"></span>
                    </span>
                    <div class="bg-white rounded-lg shadow-sm border border-blue-100 p-3 flex items-center">
                      <svg class="animate-spin -ml-1 mr-3 h-4 w-4 text-blue-500" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>
                      <p class="text-sm text-gray-600 font-mono">{{ msg.statusMsg }}</p>
                    </div>
                  </div>

                  <div v-if="msg.thinkingDetails && msg.thinkingDetails.length > 0" class="relative animate-fade-in">
                    <span class="absolute -left-[41px] flex h-5 w-5 items-center justify-center rounded-full bg-gray-100 ring-4 ring-gray-50">
                      <span class="text-xs">🧠</span>
                    </span>
                    <div class="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
                      <button @click="msg.isThinkingExpanded = !msg.isThinkingExpanded" class="w-full px-4 py-3 flex items-center justify-between bg-gray-50 hover:bg-gray-100 transition-colors">
                        <div class="flex items-center gap-3">
                          <span v-if="msg.isThinkingExpanded && isThinking" class="relative flex h-3 w-3">
                            <span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-indigo-400 opacity-75"></span>
                            <span class="relative inline-flex rounded-full h-3 w-3 bg-indigo-500"></span>
                          </span>
                          <span v-else class="text-lg">✨</span>
                          <span class="font-bold text-gray-700 text-sm">{{ msg.isThinkingExpanded ? '多智能体深度思考中...' : '查看会诊推演过程' }}</span>
                        </div>
                        <svg class="w-5 h-5 text-gray-500 transform transition-transform duration-200" :class="{'rotate-180': !msg.isThinkingExpanded}" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path></svg>
                      </button>

                      <div v-show="msg.isThinkingExpanded" class="p-4 border-t border-gray-200 bg-gray-50 space-y-4 max-h-96 overflow-y-auto">
                        <div v-for="(detail, idx) in msg.thinkingDetails" :key="'detail'+idx" class="p-4 bg-white rounded-md border border-gray-100 shadow-sm border-l-4 border-l-indigo-400">
                          <div class="flex items-center gap-2 mb-2">
                            <span class="text-xs font-bold px-2 py-1 bg-indigo-50 text-indigo-700 rounded">{{ detail.agent }}</span>
                            <span class="font-bold text-gray-800 text-sm">{{ detail.title }}</span>
                          </div>
                          <pre class="text-xs text-gray-600 whitespace-pre-wrap font-sans leading-relaxed">{{ detail.content }}<span v-if="detail.isTyping" class="inline-block w-1 h-3 ml-1 bg-indigo-400 animate-pulse"></span></pre>
                        </div>
                      </div>
                    </div>
                  </div>

                  <div v-if="msg.vision_data" class="relative animate-fade-in">
                    <span class="absolute -left-[41px] flex h-5 w-5 items-center justify-center rounded-full bg-purple-100 ring-4 ring-gray-50">
                      <span class="h-2.5 w-2.5 rounded-full bg-purple-600"></span>
                    </span>
                    <div class="bg-white rounded-xl shadow-sm border border-purple-100 p-4">
                      <p class="text-xs font-bold text-purple-600 uppercase tracking-wider mb-3">Vision Agent Output</p>
                      <div class="flex gap-4">
                        <img v-if="msg.vision_data?.visualizations?.rv_mask_base64" :src="msg.vision_data.visualizations.rv_mask_base64" class="w-1/3 rounded bg-black object-cover" />
                        <img v-if="msg.vision_data?.visualizations?.faz_mask_base64" :src="msg.vision_data.visualizations.faz_mask_base64" class="w-1/3 rounded bg-black object-cover" />
                        <div class="flex-1 bg-gray-50 rounded p-2 text-[10px] font-mono border">
                          <div v-for="(v, k) in (msg.vision_data.metrics || {})" :key="k" class="flex justify-between border-b py-1">
                            <span class="text-gray-500 truncate w-20">{{ k }}</span>
                            <span class="font-bold">{{ v }}</span>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>

                  <div v-if="msg.text" class="relative animate-fade-in">
                    <span class="absolute -left-[41px] flex h-5 w-5 items-center justify-center rounded-full bg-green-100 ring-4 ring-gray-50">
                      <span class="h-2.5 w-2.5 rounded-full bg-green-500"></span>
                    </span>
                    <div class="bg-white rounded-xl shadow-sm border border-green-100 p-4">
                       <p class="text-xs font-bold text-green-600 uppercase tracking-wider mb-2">Final Synthesis</p>
                       <p class="text-gray-800 leading-relaxed text-sm whitespace-pre-wrap">{{ msg.text }}<span v-if="isTypingFinal" class="inline-block w-1.5 h-4 ml-1 bg-green-500 animate-pulse"></span></p>
                    </div>
                  </div>

                </div>
              </div>
            </div>
          </div>
        </div>
      </main>

      <footer class="bg-white p-4 border-t border-gray-200">
        <div v-if="previewImage" class="relative inline-block mb-3 ml-2">
          <img :src="previewImage" class="h-16 rounded border border-gray-300 shadow-sm" />
          <button @click="clearImage" class="absolute -top-2 -right-2 bg-red-500 text-white rounded-full w-5 h-5 flex flex-col items-center justify-center text-xs hover:bg-red-600">×</button>
        </div>
        <div class="flex items-end bg-gray-100 rounded-2xl p-2 border border-gray-200 focus-within:border-indigo-400 transition-all">
          <label class="p-3 text-gray-400 hover:text-indigo-600 cursor-pointer transition-colors rounded-xl hover:bg-white">
            <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15.172 7l-6.586 6.586a2 2 0 102.828 2.828l6.414-6.586a4 4 0 00-5.656-5.656l-6.415 6.585a6 6 0 108.486 8.486L20.5 13"></path></svg>
            <input type="file" class="hidden" @change="handleImageSelect" accept="image/*" />
          </label>
          <textarea v-model="inputText" @keydown.enter.prevent="sendMessage" placeholder="上传图像或描述症状..." class="flex-1 bg-transparent border-0 focus:ring-0 resize-none max-h-32 min-h-[44px] py-3 px-2 text-gray-700" rows="1"></textarea>
          <button @click="sendMessage" :disabled="isThinking || (!inputText.trim() && !selectedFile)" class="p-3 ml-2 bg-indigo-600 text-white rounded-xl hover:bg-indigo-700 disabled:bg-gray-300 transition-colors shadow-sm">
            <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"></path></svg>
          </button>
        </div>
      </footer>
    </div>
  </div>
</template>

<script setup>
import { ref, nextTick, computed, onMounted } from 'vue';
import { useRouter } from 'vue-router';
import { authState, authActions } from '../stores/authStore';

const router = useRouter();
const userInitial = computed(() => authState.username ? authState.username.charAt(0).toUpperCase() : 'U');
const currentSessionId = computed(() => authState.session_id ? authState.session_id.substring(0, 8) + '...' : '等待中...');

const messages = ref([]);
const inputText = ref('');
const selectedFile = ref(null);
const previewImage = ref('');
const isThinking = ref(false);
const isTypingFinal = ref(false);
const isSending = ref(false);
const chatContainer = ref(null);

function getPersistedAuthState() {
  try {
    return JSON.parse(localStorage.getItem('heal_auth_state') || '{}');
  } catch {
    return {};
  }
}

function getAccessToken() {
  return authState.accessToken || getPersistedAuthState().accessToken || null;
}

async function ensureAuthenticated() {
  const token = getAccessToken();
  if (!token) {
    alert('登录状态失效，请重新登录');
    authActions.logout();
    router.push('/login');
    throw new Error('No access token');
  }
  return token;
}

async function ensureSessionInfo() {
  const token = await ensureAuthenticated();
  if (authState.session_id && authState.username && authState.role) {
    return authState.session_id;
  }

  const res = await fetch('http://127.0.0.1:8001/auth/me', {
    method: 'GET',
    headers: {
      Authorization: `Bearer ${token}`,
    },
  });

  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    if (res.status === 401) {
      authActions.logout();
      router.push('/login');
    }
    throw new Error(data.detail || '获取当前用户信息失败');
  }

  authState.username = data.username;
  authState.role = data.role;
  authState.session_id = data.session_id;
  return data.session_id;
}

const handleImageSelect = (e) => {
  const file = e.target.files?.[0];
  if (!file) return;
  selectedFile.value = file;
  previewImage.value = URL.createObjectURL(file);
};

const clearImage = () => {
  selectedFile.value = null;
  previewImage.value = '';
};

const scrollToBottom = async () => {
  await nextTick();
  if (chatContainer.value) chatContainer.value.scrollTop = chatContainer.value.scrollHeight;
};

const runTypewriter = (targetObj, field, fullText, speed = 15, onComplete = null) => {
  targetObj.isTyping = true;
  targetObj[field] = '';
  let i = 0;
  const timer = setInterval(() => {
    if (i < fullText.length) {
      targetObj[field] += fullText.charAt(i);
      i++;
      if (i % 5 === 0) scrollToBottom();
    } else {
      clearInterval(timer);
      targetObj.isTyping = false;
      if (onComplete) onComplete();
    }
  }, speed);
};

const sendMessage = async () => {
  if ((!inputText.value || !inputText.value.trim()) && !selectedFile.value) return;
  if (isSending.value) return;

  isSending.value = true;
  isThinking.value = true;

  let activeAgentMsg = null;

  try {
    const token = await ensureAuthenticated();
    const sessionId = await ensureSessionInfo();

    const fileSnapshot = selectedFile.value;
    const previewSnapshot = previewImage.value;
    const textSnapshot = (inputText.value || '').trim();

    messages.value.push({ role: 'user', text: textSnapshot || '开始图像分析', image: previewSnapshot || null });

    const formData = new FormData();
    formData.append('text', textSnapshot || '');
    formData.append('session_id', sessionId || '');
    if (fileSnapshot) formData.append('file', fileSnapshot);

    inputText.value = '';
    clearImage();
    await scrollToBottom();

    activeAgentMsg = ref({
      role: 'assistant',
      statusMsg: '初始化连接...',
      thinkingDetails: [],
      isThinkingExpanded: true,
      vision_data: null,
      text: ''
    });
    messages.value.push(activeAgentMsg.value);

    const response = await fetch('http://127.0.0.1:8001/api/chat', {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${token}`,
      },
      body: formData,
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new Error(err.detail || `请求失败（${response.status}）`);
    }

    if (!response.body) {
      throw new Error('服务端未返回流式响应');
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split('\n\n');
      buffer = parts.pop() || '';
      
      for (let line of parts) {
        if (!line.startsWith('data: ')) continue;
        try {
          const payload = JSON.parse(line.slice(6));
          
          if (payload.type === 'thinking') {
            activeAgentMsg.value.statusMsg = payload.msg;
          } else if (payload.type === 'thinking_detail') {
            const detail = ref({ agent: payload.agent, title: payload.title, content: '', isTyping: true });
            activeAgentMsg.value.thinkingDetails.push(detail.value);
            runTypewriter(detail.value, 'content', payload.content || '无推理细节。');
          } else if (payload.type === 'vision_data') {
            activeAgentMsg.value.vision_data = payload.data;
          } else if (payload.type === 'classify_data') {
            // 保留兼容，不额外渲染
          } else if (payload.type === 'final_text') {
            activeAgentMsg.value.statusMsg = '';
            isTypingFinal.value = true;
            runTypewriter(activeAgentMsg.value, 'text', payload.text || '', 20, () => {
              isTypingFinal.value = false;
            });
          } else if (payload.type === 'error') {
            activeAgentMsg.value.statusMsg = '';
            activeAgentMsg.value.text = `⚠️ ${payload.msg || '服务异常'}`;
            isThinking.value = false;
          } else if (payload.type === 'done') {
            isThinking.value = false;
            setTimeout(() => { activeAgentMsg.value.isThinkingExpanded = false; }, 1500);
          }
          scrollToBottom();
        } catch (err) {
          console.error('SSE Error:', err);
        }
      }
    }
  } catch (error) {
    console.error(error);
    if (activeAgentMsg?.value) {
      activeAgentMsg.value.statusMsg = '';
      activeAgentMsg.value.text = `⚠️ ${error.message || '连接异常，请重试。'}`;
    } else {
      messages.value.push({
        role: 'assistant',
        statusMsg: '',
        thinkingDetails: [],
        isThinkingExpanded: false,
        vision_data: null,
        text: `⚠️ ${error.message || '连接异常，请重试。'}`,
      });
    }
    isThinking.value = false;
  } finally {
    isSending.value = false;
    await scrollToBottom();
  }
};

const handleLogout = () => {
  if (confirm('退出登录？')) {
    authActions.logout();
    router.push('/login');
  }
};

onMounted(async () => {
  try {
    await ensureSessionInfo();
  } catch (e) {
    console.warn('初始化会话失败：', e.message);
  }
});
</script>

<style scoped>
.animate-fade-in { animation: fadeIn 0.5s ease-out forwards; }
@keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 3px; }
</style>
