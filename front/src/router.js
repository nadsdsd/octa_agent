import { createRouter, createWebHistory } from 'vue-router';
import { authState } from './stores/authStore';

const routes = [
  {
    path: '/',
    redirect: '/login' // 默认访问重定向到登录页
  },
  {
    path: '/login',
    name: 'Login',
    component: () => import('./components/Login.vue'),
    meta: { requiresAuth: false } 
  },
  {
    path: '/register',
    name: 'Register',
    component: () => import('./components/Register.vue'),
    meta: { requiresAuth: false }
  },
  {
    path: '/chat',
    name: 'Chat',
    // 这里指向你现有的对话界面组件
    component: () => import('./components/ChatAgent.vue'),
    meta: { requiresAuth: true } // 标记为需要登录才能访问
  }
];

const router = createRouter({
  history: createWebHistory(),
  routes,
});

// 全局路由守卫：拦截非法跳转
router.beforeEach((to, from, next) => {
  const isAuthenticated = authState.isAuthenticated;

  if (to.meta.requiresAuth && !isAuthenticated) {
    // 1. 去需要登录的页面，但没登录 -> 踢回登录页
    next({ name: 'Login' });
  } else if (!to.meta.requiresAuth && isAuthenticated && (to.name === 'Login' || to.name === 'Register')) {
    // 2. 已经登录了，还想去登录/注册页 -> 直接送去聊天页
    next({ name: 'Chat' });
  } else {
    // 3. 正常放行
    next();
  }
});

export default router;