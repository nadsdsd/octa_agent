<template>
  <div class="auth-wrapper">
    <div class="auth-card">
      <h2>登录 Heal Agent</h2>
      <form @submit.prevent="handleLogin">
        <div class="input-group">
          <label>用户名</label>
          <input v-model="username" type="text" required placeholder="请输入用户名" />
        </div>
        <div class="input-group">
          <label>密码</label>
          <input v-model="password" type="password" required placeholder="请输入密码" />
        </div>
        
        <div v-if="error" class="error-msg">{{ error }}</div>
        
        <button type="submit" class="submit-btn" :disabled="loading">
          {{ loading ? '登录中...' : '登 录' }}
        </button>
      </form>
      
      <div class="switch-link">
        还没有账号？<a href="#" @click.prevent="$router.push('/register')">去注册</a>
      </div>
    </div>
  </div>
</template>

<script>
import { ref } from 'vue';
import { useRouter } from 'vue-router';
import { authActions } from '../stores/authStore';

export default {
  setup() {
    const router = useRouter();
    const username = ref('');
    const password = ref('');
    const loading = ref(false);
    const error = ref('');

    const handleLogin = async () => {
      error.value = '';
      loading.value = true;
      try {
        await authActions.login(username.value, password.value);
        // 登录成功，跳转到聊天主界面
        router.push({ name: 'Chat' });
      } catch (err) {
        error.value = err.message;
      } finally {
        loading.value = false;
      }
    };

    return { username, password, loading, error, handleLogin };
  }
}
</script>

<style scoped>
.auth-wrapper {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 100vh;
}
.auth-card {
  background: white;
  padding: 40px;
  border-radius: 10px;
  box-shadow: 0 4px 12px rgba(0,0,0,0.1);
  width: 350px;
}
h2 { text-align: center; color: #333; margin-bottom: 25px; }
.input-group { margin-bottom: 20px; }
.input-group label { display: block; margin-bottom: 8px; color: #666; font-size: 14px;}
.input-group input { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 6px; box-sizing: border-box;}
.submit-btn { width: 100%; padding: 12px; background-color: #4CAF50; color: white; border: none; border-radius: 6px; cursor: pointer; font-size: 16px;}
.submit-btn:disabled { background-color: #9E9E9E; }
.error-msg { color: #f44336; font-size: 13px; margin-bottom: 15px; text-align: center;}
.switch-link { text-align: center; margin-top: 20px; font-size: 14px; color: #666;}
.switch-link a { color: #4CAF50; text-decoration: none; }
</style>