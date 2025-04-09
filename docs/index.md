---
# https://vitepress.dev/reference/default-theme-home-page
layout: home

hero:
  name: "<span class='hero-name'>码医森</span>"  # 这里添加了 span 标签
  text: "<span class='hero-text'>Ethan的知识博客</span>"
  tagline: 涵盖各类计算机技术；个人其他能力和认知提升；多学科探索自我
  image:
    src: /imgs/home-page-logo.svg
  
  actions:
    - theme: brand
      text: 面经🔎
      link: /Job_Interview/
    - theme: alt
      text: 站点更新📔
      link: /update/update_log
    - theme: alt
      text: 关于站长🙈
      link: /about_me/

features:
  - icon:
      src: /icons/learn.svg
    title: IT学习
    details: 涵盖各类计算机知识体系。408知识点、Java后端、Linux等
    link: '/IT-learning/'
  - icon: 
      src: /icons/improve.svg
    title: 我的感悟
    details: 除了IT，还有一些更nice的模块。比如投资, 阅读等
    link: '/my_think/'
  - icon: 
      src: /icons/explore.svg
    title: 人工智能
    details: 算法学习, 高性能, 对当下热点的分析。
    link: '/AI/'
---

<style>
:root {
  --vp-home-hero-name-color: transparent;
  --vp-home-hero-name-background: -webkit-linear-gradient(120deg, #bd34fe 30%, #41d1ff);

  --vp-home-hero-image-background-image: linear-gradient(-45deg, #bd34fe 50%, #47caff 50%);
  --vp-home-hero-image-filter: blur(44px);
}

@media (min-width: 640px) {
  :root {
    --vp-home-hero-image-filter: blur(56px);
  }
}

@media (min-width: 960px) {
  :root {
    --vp-home-hero-image-filter: blur(68px);
  }
}
</style>
