<template>
  <div class="chat-page">
    <aside class="sidebar">
      <div class="brand">
        <div class="brand-logo">AI</div>
        <div>
          <div class="brand-title">OCTA 智能诊断助手</div>
          <div class="brand-subtitle">眼科自检 / 辅助分析</div>
        </div>
      </div>

      <div class="user-card">
        <div class="avatar">{{ userInitial }}</div>
        <div class="user-meta">
          <div class="username">{{ authState.username || '未登录' }}</div>
          <div class="role">{{ authState.role || 'guest' }}</div>
        </div>
      </div>

      <div class="session-box">
        <div class="session-label">当前会话</div>
        <div class="session-value">{{ displaySessionId }}</div>
      </div>

      <button class="logout-btn" @click="handleLogout">退出登录</button>
    </aside>

    <main class="chat-main">
      <header class="chat-header">
        <div>
          <h1>眼底影像智能分析</h1>
          <p>上传 OCTA 图像，结合多阶段工作流完成分割、诊断、会诊与建议生成。</p>
        </div>
      </header>

      <section ref="chatContainer" class="chat-container">
        <div v-if="messages.length === 0" class="empty-state">
          <div class="empty-icon">👁️</div>
          <div class="empty-title">开始一次新的诊断会话</div>
          <div class="empty-desc">
            你可以直接输入问题，或者上传 OCTA 图像后点击发送。
          </div>
        </div>

        <div
          v-for="(msg, index) in messages"
          :key="index"
          :class="['message-row', msg.role === 'user' ? 'user-row' : 'assistant-row']"
        >
          <div class="bubble" :class="msg.role === 'user' ? 'user-bubble' : 'assistant-bubble'">
            <template v-if="msg.role === 'user'">
              <div v-if="msg.image" class="user-image-wrap">
                <img :src="msg.image" alt="uploaded" class="user-image" />
              </div>
              <div class="message-text">{{ msg.text }}</div>
            </template>

            <template v-else>
              <div v-if="msg.statusMsg" class="status-msg">
                <span class="thinking-dot" v-if="isThinking">●</span>
                {{ msg.statusMsg }}
              </div>

              <div v-if="msg.vision_data" class="vision-card">
                <div class="vision-title">视觉分析结果</div>
                <div class="vision-grid">
                  <div class="vision-item">
                    <span class="k">扫描类型</span>
                    <span class="v">{{ msg.vision_data?.image_metadata?.scan_type || '未知' }}</span>
                  </div>
                  <div class="vision-item">
                    <span class="k">图像有效性</span>
                    <span class="v">
                      {{
                        msg.vision_data?.image_metadata?.is_valid_octa
                          ? '有效 OCTA'
                          : '无效 / 质量异常'
                      }}
                    </span>
                  </div>
                </div>
              </div>

              <div
                v-if="msg.thinkingDetails && msg.thinkingDetails.length"
                class="thinking-panel"
              >
                <div class="thinking-header" @click="msg.isThinkingExpanded = !msg.isThinkingExpanded">
                  <span>推理过程</span>
                  <span>{{ msg.isThinkingExpanded ? '收起' : '展开' }}</span>
                </div>

                <div v-show="msg.isThinkingExpanded" class="thinking-list">
                  <div
                    v-for="(detail, detailIndex) in msg.thinkingDetails"
                    :key="detailIndex"
                    class="thinking-item"
                  >
                    <div class="thinking-agent">{{ detail.agent }}</div>
                    <div class="thinking-title">{{ detail.title }}</div>
                    <div class="thinking-content">{{ detail.content }}</div>
                  </div>
                </div>
              </div>

              <div v-if="msg.text" class="assistant-text">
                {{ msg.text }}
              </div>
            </template>
          </div>
        </div>
      </section>

      <section class="composer">
        <div v-if="previewImage" class="preview-wrap">
          <img :src="previewImage" alt="preview" class="preview-image" />
          <button class="remove-image-btn" @click="clearImage">移除图片</button>
        </div>

        <div class="composer-row">
          <label class="upload-btn">
            <input type="file" accept="image/*" hidden @change="handleImageSelect" />
            上传图像
          </label>

          <input
            v-model="inputText"
            class="text-input"
            type="text"
            placeholder="请输入你的问题，例如：请帮我分析这张图，或给我诊断建议"
            @keydown.enter="sendMessage"
          />

          <button class="send-btn" :disabled="isSending" @click="sendMessage">
            {{ isSending ? '发送中...' : '发送' }}
          </button>
        </div>
      </section>
    </main>
  </div>
</template>

<script setup>
import { ref, nextTick, computed, onMounted } from 'vue';
import { useRouter } from 'vue-router';
import { authState, authActions } from '../stores/authStore';

const router = useRouter();

const messages = ref([]);
const inputText = ref('');
const selectedFile = ref(null);
const previewImage = ref('');
const isThinking = ref(false);
const isSending = ref(false);
const chatContainer = ref(null);

const userInitial = computed(() => {
  const name = authState.username || 'U';
  return name.charAt(0).toUpperCase();
});

const displaySessionId = computed(() => {
  if (!authState.session_id) return '未初始化';
  return authState.session_id.length > 20
    ? `${authState.session_id.slice(0, 20)}...`
    : authState.session_id;
});

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

  // 如果 authStore 里已经有 session_id，就不重复查
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
  if (chatContainer.value) {
    chatContainer.value.scrollTop = chatContainer.value.scrollHeight;
  }
};

const runTypewriter = (targetObj, field, fullText, speed = 15, onComplete = null) => {
  targetObj.isTyping = true;
  targetObj[field] = '';
  let i = 0;

  const timer = setInterval(() => {
    if (i < fullText.length) {
      targetObj[field] += fullText.charAt(i);
      i++;
      if (i % 4 === 0) scrollToBottom();
    } else {
      clearInterval(timer);
      targetObj.isTyping = false;
      onComplete?.();
    }
  }, speed);
};

async function sendMessage() {
  if ((!inputText.value || !inputText.value.trim()) && !selectedFile.value) return;
  if (isSending.value) return;

  isSending.value = true;
  isThinking.value = true;

  try {
    const token = await ensureAuthenticated();
    const sessionId = await ensureSessionInfo();

    messages.value.push({
      role: 'user',
      text: inputText.value?.trim() || '开始图像分析',
      image: previewImage.value || null,
    });

    const currentText = inputText.value;
    inputText.value = '';
    clearImage();
    await scrollToBottom();

    const assistantMsg = {
      role: 'assistant',
      statusMsg: '初始化连接...',
      thinkingDetails: [],
      isThinkingExpanded: true,
      vision_data: null,
      text: '',
      isTyping: false,
    };
    messages.value.push(assistantMsg);
    await scrollToBottom();

    const formData = new FormData();
    formData.append('text', currentText || '');
    formData.append('session_id', sessionId || '');
    if (selectedFile.value) {
      formData.append('file', selectedFile.value);
    }

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

      for (const block of parts) {
        const line = block.trim();
        if (!line.startsWith('data: ')) continue;

        let payload;
        try {
          payload = JSON.parse(line.slice(6));
        } catch (e) {
          console.error('SSE JSON parse failed:', e, line);
          continue;
        }

        if (payload.type === 'thinking') {
          assistantMsg.statusMsg = payload.msg || '处理中...';
        } else if (payload.type === 'thinking_detail') {
          const detail = {
            agent: payload.agent || 'Agent',
            title: payload.title || '推理步骤',
            content: '',
            isTyping: true,
          };
          assistantMsg.thinkingDetails.push(detail);
          runTypewriter(detail, 'content', payload.content || '');
        } else if (payload.type === 'vision_data') {
          assistantMsg.vision_data = payload.data;
        } else if (payload.type === 'warning') {
          assistantMsg.thinkingDetails.push({
            agent: 'System',
            title: '警告',
            content: payload.msg || '发生警告',
            isTyping: false,
          });
        } else if (payload.type === 'final_text') {
          assistantMsg.statusMsg = '';
          runTypewriter(assistantMsg, 'text', payload.text || '', 18, () => {
            assistantMsg.isThinkingExpanded = false;
          });
        } else if (payload.type === 'error') {
          assistantMsg.statusMsg = '';
          assistantMsg.text = `⚠️ ${payload.msg || '服务异常'}`;
        } else if (payload.type === 'done') {
          isThinking.value = false;
          if (!assistantMsg.text && !assistantMsg.statusMsg) {
            assistantMsg.statusMsg = '处理完成';
          }
        }

        await scrollToBottom();
      }
    }
  } catch (error) {
    console.error(error);
    messages.value.push({
      role: 'assistant',
      statusMsg: '',
      thinkingDetails: [],
      isThinkingExpanded: false,
      vision_data: null,
      text: `⚠️ ${error.message || '连接异常，请重试。'}`,
      isTyping: false,
    });
  } finally {
    isSending.value = false;
    isThinking.value = false;
    await scrollToBottom();
  }
}

function handleLogout() {
  if (!confirm('确定退出登录吗？')) return;
  authActions.logout();
  router.push('/login');
}

onMounted(async () => {
  try {
    await ensureSessionInfo();
  } catch (e) {
    console.warn('初始化会话失败：', e.message);
  }
});
</script>

<style scoped>
.chat-page {
  display: flex;
  min-height: 100vh;
  background: #f5f7fb;
  color: #1f2937;
}

.sidebar {
  width: 280px;
  background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
  color: #fff;
  padding: 24px 20px;
  display: flex;
  flex-direction: column;
  gap: 20px;
  box-sizing: border-box;
}

.brand {
  display: flex;
  align-items: center;
  gap: 12px;
}

.brand-logo {
  width: 44px;
  height: 44px;
  border-radius: 14px;
  background: #6366f1;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: 700;
}

.brand-title {
  font-size: 16px;
  font-weight: 700;
}

.brand-subtitle {
  font-size: 12px;
  opacity: 0.8;
}

.user-card,
.session-box {
  background: rgba(255, 255, 255, 0.08);
  border-radius: 16px;
  padding: 14px;
}

.user-card {
  display: flex;
  gap: 12px;
  align-items: center;
}

.avatar {
  width: 42px;
  height: 42px;
  border-radius: 50%;
  background: #818cf8;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: 700;
}

.username {
  font-weight: 600;
}

.role,
.session-label,
.session-value {
  font-size: 12px;
  opacity: 0.85;
  word-break: break-all;
}

.logout-btn {
  margin-top: auto;
  border: none;
  border-radius: 12px;
  padding: 12px 14px;
  background: #ef4444;
  color: #fff;
  cursor: pointer;
  font-weight: 600;
}

.chat-main {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-width: 0;
}

.chat-header {
  padding: 24px 28px 8px;
}

.chat-header h1 {
  margin: 0 0 8px;
  font-size: 24px;
}

.chat-header p {
  margin: 0;
  color: #6b7280;
}

.chat-container {
  flex: 1;
  overflow-y: auto;
  padding: 16px 28px 120px;
  box-sizing: border-box;
}

.empty-state {
  height: 100%;
  min-height: 420px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  color: #6b7280;
  text-align: center;
}

.empty-icon {
  font-size: 48px;
  margin-bottom: 16px;
}

.empty-title {
  font-size: 20px;
  font-weight: 700;
  margin-bottom: 8px;
}

.message-row {
  display: flex;
  margin-bottom: 18px;
}

.user-row {
  justify-content: flex-end;
}

.assistant-row {
  justify-content: flex-start;
}

.bubble {
  max-width: min(920px, 82%);
  border-radius: 18px;
  padding: 14px 16px;
  box-sizing: border-box;
  white-space: pre-wrap;
  word-break: break-word;
}

.user-bubble {
  background: #4f46e5;
  color: #fff;
}

.assistant-bubble {
  background: #fff;
  color: #111827;
  border: 1px solid #e5e7eb;
}

.user-image-wrap {
  margin-bottom: 10px;
}

.user-image,
.preview-image {
  max-width: 220px;
  border-radius: 14px;
  display: block;
}

.message-text,
.assistant-text {
  line-height: 1.7;
}

.status-msg {
  color: #4b5563;
  margin-bottom: 10px;
  font-size: 14px;
}

.thinking-dot {
  margin-right: 8px;
  color: #6366f1;
}

.thinking-panel {
  margin-top: 10px;
  border: 1px solid #e5e7eb;
  border-radius: 14px;
  overflow: hidden;
  background: #fafafa;
}

.thinking-header {
  display: flex;
  justify-content: space-between;
  padding: 10px 12px;
  cursor: pointer;
  background: #f3f4f6;
  font-size: 13px;
  font-weight: 600;
}

.thinking-list {
  padding: 10px 12px;
}

.thinking-item + .thinking-item {
  margin-top: 12px;
  padding-top: 12px;
  border-top: 1px dashed #d1d5db;
}

.thinking-agent {
  font-size: 12px;
  color: #6366f1;
  font-weight: 700;
  margin-bottom: 4px;
}

.thinking-title {
  font-size: 13px;
  font-weight: 700;
  margin-bottom: 6px;
}

.thinking-content {
  font-size: 14px;
  line-height: 1.7;
  color: #374151;
}

.vision-card {
  margin: 10px 0;
  padding: 12px;
  background: #eef2ff;
  border-radius: 12px;
}

.vision-title {
  font-weight: 700;
  margin-bottom: 10px;
}

.vision-grid {
  display: grid;
  gap: 8px;
}

.vision-item {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  font-size: 14px;
}

.vision-item .k {
  color: #6b7280;
}

.vision-item .v {
  font-weight: 600;
}

.composer {
  position: sticky;
  bottom: 0;
  padding: 16px 28px 20px;
  background: linear-gradient(180deg, rgba(245,247,251,0) 0%, #f5f7fb 18%, #f5f7fb 100%);
}

.preview-wrap {
  display: inline-flex;
  flex-direction: column;
  gap: 8px;
  margin-bottom: 12px;
}

.remove-image-btn {
  align-self: flex-start;
  border: none;
  background: #ef4444;
  color: #fff;
  padding: 8px 10px;
  border-radius: 10px;
  cursor: pointer;
}

.composer-row {
  display: flex;
  gap: 12px;
  align-items: center;
  background: #fff;
  border: 1px solid #e5e7eb;
  padding: 12px;
  border-radius: 18px;
}

.upload-btn,
.send-btn {
  border: none;
  border-radius: 12px;
  padding: 12px 16px;
  cursor: pointer;
  font-weight: 600;
}

.upload-btn {
  background: #e0e7ff;
  color: #3730a3;
}

.send-btn {
  background: #4f46e5;
  color: #fff;
  min-width: 96px;
}

.send-btn:disabled {
  opacity: 0.7;
  cursor: not-allowed;
}

.text-input {
  flex: 1;
  border: none;
  outline: none;
  background: transparent;
  font-size: 15px;
}

@media (max-width: 960px) {
  .chat-page {
    flex-direction: column;
  }

  .sidebar {
    width: 100%;
  }

  .bubble {
    max-width: 100%;
  }
}
</style>