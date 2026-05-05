import { reactive, watch } from 'vue';

const AUTH_BACKEND_URL = 'http://127.0.0.1:8000';
const CHAT_BACKEND_URL = 'http://127.0.0.1:8001';

const storedState = JSON.parse(localStorage.getItem('heal_auth_state') || 'null') || {
  isAuthenticated: false,
  accessToken: null,
  username: null,
  role: null,
  session_id: null,
};

export const authState = reactive(storedState);

watch(
  authState,
  (newState) => {
    localStorage.setItem('heal_auth_state', JSON.stringify(newState));
  },
  { deep: true }
);

async function fetch8001SessionInfo(accessToken) {
  const res = await fetch(`${CHAT_BACKEND_URL}/auth/me`, {
    method: 'GET',
    headers: {
      Authorization: `Bearer ${accessToken}`,
    },
  });

  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(data.detail || '获取 8001 会话信息失败');
  }
  return data;
}

export const authActions = {
  async register(username, password, role = 'doctor') {
    const res = await fetch(`${AUTH_BACKEND_URL}/api/auth/register`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password, role }),
    });

    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || '注册失败');

    return data;
  },

  async login(username, password) {
    const res = await fetch(`${AUTH_BACKEND_URL}/api/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password }),
    });

    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || '用户名或密码错误');

    authState.isAuthenticated = true;
    authState.accessToken = data.access_token;
    authState.username = data.username;
    authState.role = data.role;
    authState.session_id = null;

    const me = await fetch8001SessionInfo(data.access_token);
    authState.session_id = me.session_id || null;

    return data;
  },

  async refreshSessionInfo() {
    if (!authState.accessToken) return null;
    const me = await fetch8001SessionInfo(authState.accessToken);
    authState.username = me.username ?? authState.username;
    authState.role = me.role ?? authState.role;
    authState.session_id = me.session_id ?? authState.session_id;
    return me;
  },

  logout() {
    authState.isAuthenticated = false;
    authState.accessToken = null;
    authState.username = null;
    authState.role = null;
    authState.session_id = null;
    localStorage.removeItem('heal_auth_state');
  },
};
