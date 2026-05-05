<template>
  <div class="auth-wrapper">
    <div class="auth-card">
      <h2>注册账号</h2>
      <form @submit.prevent="handleRegister">
        <div class="input-group">
          <label>用户名</label>
          <input v-model="username" type="text" required placeholder="设置你的用户名" />
        </div>
        <div class="input-group">
          <label>密码</label>
          <input v-model="password" type="password" required placeholder="设置密码" />
        </div>
        <div class="input-group">
          <label>身份</label>
          <select v-model="role" class="role-select">
            <option value="doctor">医生</option>
            <option value="user">普通患者</option>
          </select>
        </div>
        
        <div v-if="error" class="error-msg">{{ error }}</div>
        <div v-if="successMsg" class="success-msg">{{ successMsg }}</div>
        
        <button type="submit" class="submit-btn" :disabled="loading">
          {{ loading ? '提交中...' : '注 册' }}
        </button>
      </form>
      
      <div class="switch-link">
        已有账号？<a href="#" @click.prevent="$router.push('/login')">去登录</a>
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
    const role = ref('doctor');
    const loading = ref(false);
    const error = ref('');
    const successMsg = ref('');

    const handleRegister = async () => {
      error.value = '';
      successMsg.value = '';
      loading.value = true;
      try {
        await authActions.register(username.value, password.value, role.value);
        successMsg.value = '注册成功！正在跳转登录页...';
        setTimeout(() => {
          router.push('/login'); // 注册成功后跳回登录页
        }, 1500);
      } catch (err) {
        error.value = err.message;
      } finally {
        loading.value = false;
      }
    };

    return { username, password, role, loading, error, successMsg, handleRegister };
  }
}
</script>

<style scoped>
/* 样式与登录页基本一致，这里省略重复的 CSS，只需添加 select 的样式 */
.auth-wrapper { display: flex; justify-content: center; align-items: center; height: 100vh; }
.auth-card { background: white; padding: 40px; border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); width: 350px; }
h2 { text-align: center; color: #333; margin-bottom: 25px; }
.input-group { margin-bottom: 20px; }
.input-group label { display: block; margin-bottom: 8px; color: #666; font-size: 14px;}
.input-group input, .role-select { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 6px; box-sizing: border-box;}
.submit-btn { width: 100%; padding: 12px; background-color: #4CAF50; color: white; border: none; border-radius: 6px; cursor: pointer; font-size: 16px;}
.submit-btn:disabled { background-color: #9E9E9E; }
.error-msg { color: #f44336; font-size: 13px; margin-bottom: 15px; text-align: center;}
.success-msg { color: #4CAF50; font-size: 13px; margin-bottom: 15px; text-align: center;}
.switch-link { text-align: center; margin-top: 20px; font-size: 14px; color: #666;}
.switch-link a { color: #4CAF50; text-decoration: none; }
</style>