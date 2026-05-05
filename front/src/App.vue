<template>
  <RouterView />
</template>

<script>
export default {
  name: 'App'
}
</script>

<style>
/* 全局样式，去掉浏览器默认的边距 */
body, html {
  margin: 0;
  padding: 0;
  height: 100vh;
  background-color: #f5f7fa;
}
</style>