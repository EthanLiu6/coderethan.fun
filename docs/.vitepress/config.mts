import { defineConfig } from 'vitepress'
import AutoSidebar from "vite-plugin-vitepress-auto-sidebar";
import markdownItKatex from 'markdown-it-katex'

const customElements = [
  'math',
  'maction',
  'maligngroup',
  'malignmark',
  'menclose',
  'merror',
  'mfenced',
  'mfrac',
  'mi',
  'mlongdiv',
  'mmultiscripts',
  'mn',
  'mo',
  'mover',
  'mpadded',
  'mphantom',
  'mroot',
  'mrow',
  'ms',
  'mscarries',
  'mscarry',
  'mscarries',
  'msgroup',
  'mstack',
  'mlongdiv',
  'msline',
  'mstack',
  'mspace',
  'msqrt',
  'msrow',
  'mstack',
  'mstack',
  'mstyle',
  'msub',
  'msup',
  'msubsup',
  'mtable',
  'mtd',
  'mtext',
  'mtr',
  'munder',
  'munderover',
  'semantics',
  'math',
  'mi',
  'mn',
  'mo',
  'ms',
  'mspace',
  'mtext',
  'menclose',
  'merror',
  'mfenced',
  'mfrac',
  'mpadded',
  'mphantom',
  'mroot',
  'mrow',
  'msqrt',
  'mstyle',
  'mmultiscripts',
  'mover',
  'mprescripts',
  'msub',
  'msubsup',
  'msup',
  'munder',
  'munderover',
  'none',
  'maligngroup',
  'malignmark',
  'mtable',
  'mtd',
  'mtr',
  'mlongdiv',
  'mscarries',
  'mscarry',
  'msgroup',
  'msline',
  'msrow',
  'mstack',
  'maction',
  'semantics',
  'annotation',
  'annotation-xml'
]

// https://vitepress.dev/reference/site-config
export default defineConfig({
  title: "码医森",
  description: "计算机知识的学习站点",
  head: [
    ['link', { rel: 'icon', type: 'image/svg+xml', href: '/imgs/home-page-logo.svg' }], // 设置 favicon
    ['link', { rel: 'stylesheet', href: 'https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.5.1/katex.min.css', crossorigin: '' }]  // 引入Katex
  ],

  lang: 'zh-CN',
  // appearance: "dark", //考虑到阅读者的护眼和博客的简约，不搞那些绚丽了
  srcExclude: ['/README.md'],

  // DeadLinks ignore
  // ignoreDeadLinks: true,

  // 自动侧边栏配置
  vite: {
    plugins: [
      // add plugin
      AutoSidebar({
        ignoreList: ["README.md"], // 忽略文件夹
        path: "/docs/", // 侧边栏扫描路径(也就是所有笔记所在的目录)
        ignoreIndexItem: true, // 忽略首页
        collapsed: false, // 是否启用折叠，默认为false展开
        deletePrefix: "docs", // 删除路径前缀
        sideBarResolved(data) {
          // 接收完整的侧边栏对象以进行自定义修改
          return data;
        },
        sideBarItemsResolved(data) {
          // 接收完整的侧边栏 subItem 对象以进行自定义修改
          return data;
        },
        beforeCreateSideBarItems(data) {
          // 获取生成侧边栏子项之前扫描的文件名列表。如果要对侧边栏数据进行排序，建议使用
          return data;
        },
        titleFromFile: false, // 从文件中提取标题
        // You can also set options to adjust sidebar data
        // see option document below
      }),
    ],
  },

  // Katex
  markdown: {
    config: (md) => {
      md.use(markdownItKatex)
    }
  },
  vue: {
    template: {
      compilerOptions: {
        isCustomElement: (tag) => customElements.includes(tag)
      }
    }
  },

  themeConfig: {
    // https://vitepress.dev/reference/default-theme-config
    siteTitle: 'CoderEthan学习站', // 左上角的站点名称
    logo: '/imgs/home-page-logo.svg',
    outline: {
      label: '本文目录', // 将 "On This Page" 改为 "目录"
      level: [2, 4]
    },
    // 启用本地搜索
    search: {
      provider: 'local'
    },

    // 导航栏旁边社交账号
    socialLinks: [
      {
        icon: { svg: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 496 512"><!--!Font Awesome Free 6.6.0 by @fontawesome - https://fontawesome.com License - https://fontawesome.com/license/free Copyright 2024 Fonticons, Inc.--><path fill="#6841d2" d="M165.9 397.4c0 2-2.3 3.6-5.2 3.6-3.3 .3-5.6-1.3-5.6-3.6 0-2 2.3-3.6 5.2-3.6 3-.3 5.6 1.3 5.6 3.6zm-31.1-4.5c-.7 2 1.3 4.3 4.3 4.9 2.6 1 5.6 0 6.2-2s-1.3-4.3-4.3-5.2c-2.6-.7-5.5 .3-6.2 2.3zm44.2-1.7c-2.9 .7-4.9 2.6-4.6 4.9 .3 2 2.9 3.3 5.9 2.6 2.9-.7 4.9-2.6 4.6-4.6-.3-1.9-3-3.2-5.9-2.9zM244.8 8C106.1 8 0 113.3 0 252c0 110.9 69.8 205.8 169.5 239.2 12.8 2.3 17.3-5.6 17.3-12.1 0-6.2-.3-40.4-.3-61.4 0 0-70 15-84.7-29.8 0 0-11.4-29.1-27.8-36.6 0 0-22.9-15.7 1.6-15.4 0 0 24.9 2 38.6 25.8 21.9 38.6 58.6 27.5 72.9 20.9 2.3-16 8.8-27.1 16-33.7-55.9-6.2-112.3-14.3-112.3-110.5 0-27.5 7.6-41.3 23.6-58.9-2.6-6.5-11.1-33.3 2.6-67.9 20.9-6.5 69 27 69 27 20-5.6 41.5-8.5 62.8-8.5s42.8 2.9 62.8 8.5c0 0 48.1-33.6 69-27 13.7 34.7 5.2 61.4 2.6 67.9 16 17.7 25.8 31.5 25.8 58.9 0 96.5-58.9 104.2-114.8 110.5 9.2 7.9 17 22.9 17 46.4 0 33.7-.3 75.4-.3 83.6 0 6.5 4.6 14.4 17.3 12.1C428.2 457.8 496 362.9 496 252 496 113.3 383.5 8 244.8 8zM97.2 352.9c-1.3 1-1 3.3 .7 5.2 1.6 1.6 3.9 2.3 5.2 1 1.3-1 1-3.3-.7-5.2-1.6-1.6-3.9-2.3-5.2-1zm-10.8-8.1c-.7 1.3 .3 2.9 2.3 3.9 1.6 1 3.6 .7 4.3-.7 .7-1.3-.3-2.9-2.3-3.9-2-.6-3.6-.3-4.3 .7zm32.4 35.6c-1.6 1.3-1 4.3 1.3 6.2 2.3 2.3 5.2 2.6 6.5 1 1.3-1.3 .7-4.3-1.3-6.2-2.2-2.3-5.2-2.6-6.5-1zm-11.4-14.7c-1.6 1-1.6 3.6 0 5.9 1.6 2.3 4.3 3.3 5.6 2.3 1.6-1.3 1.6-3.9 0-6.2-1.4-2.3-4-3.3-5.6-2z"/></svg>' },
        link: 'https://github.com/ethanliu6/'
      },
      // { icon: {svg: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 448 512"><!--!Font Awesome Free 6.6.0 by @fontawesome - https://fontawesome.com License - https://fontawesome.com/license/free Copyright 2024 Fonticons, Inc.--><path fill="#B197FC" d="M96 0C43 0 0 43 0 96L0 416c0 53 43 96 96 96l288 0 32 0c17.7 0 32-14.3 32-32s-14.3-32-32-32l0-64c17.7 0 32-14.3 32-32l0-320c0-17.7-14.3-32-32-32L384 0 96 0zm0 384l256 0 0 64L96 448c-17.7 0-32-14.3-32-32s14.3-32 32-32zm32-240c0-8.8 7.2-16 16-16l192 0c8.8 0 16 7.2 16 16s-7.2 16-16 16l-192 0c-8.8 0-16-7.2-16-16zm16 48l192 0c8.8 0 16 7.2 16 16s-7.2 16-16 16l-192 0c-8.8 0-16-7.2-16-16s7.2-16 16-16z"/></svg>'},
      {
        icon: { svg: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512"><!--!Font Awesome Free 6.6.0 by @fontawesome - https://fontawesome.com License - https://fontawesome.com/license/free Copyright 2024 Fonticons, Inc.--><path fill="#6841d2" d="M488.6 104.1C505.3 122.2 513 143.8 511.9 169.8V372.2C511.5 398.6 502.7 420.3 485.4 437.3C468.2 454.3 446.3 463.2 419.9 464H92C65.6 463.2 43.8 454.2 26.7 436.8C9.7 419.4 .8 396.5 0 368.2V169.8C.8 143.8 9.7 122.2 26.7 104.1C43.8 87.8 65.6 78.8 92 78H121.4L96.1 52.2C90.3 46.5 87.4 39.2 87.4 30.4C87.4 21.6 90.3 14.3 96.1 8.6C101.8 2.9 109.1 0 117.9 0C126.7 0 134 2.9 139.8 8.6L213.1 78H301.1L375.6 8.6C381.7 2.9 389.2 0 398 0C406.8 0 414.1 2.9 419.9 8.6C425.6 14.3 428.5 21.6 428.5 30.4C428.5 39.2 425.6 46.5 419.9 52.2L394.6 78L423.9 78C450.3 78.8 471.9 87.8 488.6 104.1H488.6zM449.8 173.8C449.4 164.2 446.1 156.4 439.1 150.3C433.9 144.2 425.1 140.9 416.4 140.5H96.1C86.5 140.9 78.6 144.2 72.5 150.3C66.3 156.4 63.1 164.2 62.7 173.8V368.2C62.7 377.4 66 385.2 72.5 391.7C79 398.2 86.9 401.5 96.1 401.5H416.4C425.6 401.5 433.4 398.2 439.7 391.7C446 385.2 449.4 377.4 449.8 368.2L449.8 173.8zM185.5 216.5C191.8 222.8 195.2 230.6 195.6 239.7V273C195.2 282.2 191.9 289.9 185.8 296.2C179.6 302.5 171.8 305.7 162.2 305.7C152.6 305.7 144.7 302.5 138.6 296.2C132.5 289.9 129.2 282.2 128.8 273V239.7C129.2 230.6 132.6 222.8 138.9 216.5C145.2 210.2 152.1 206.9 162.2 206.5C171.4 206.9 179.2 210.2 185.5 216.5H185.5zM377 216.5C383.3 222.8 386.7 230.6 387.1 239.7V273C386.7 282.2 383.4 289.9 377.3 296.2C371.2 302.5 363.3 305.7 353.7 305.7C344.1 305.7 336.3 302.5 330.1 296.2C323.1 289.9 320.7 282.2 320.4 273V239.7C320.7 230.6 324.1 222.8 330.4 216.5C336.7 210.2 344.5 206.9 353.7 206.5C362.9 206.9 370.7 210.2 377 216.5H377z"/></svg>' },
        link: 'https://space.bilibili.com/1327099977/'
      }
    ], // end社交账号

    // 顶部导航栏
    nav: [
      { text: '主页', link: '/' },
      { text: '博客指南', link: '/guide/' },
      { text: '生活与算法', link: '/生活与算法/' },
      { text: 'AI', link: '/AI/' },
      {
        text: '计算机学科内容',  // 合并J为一个大类
        items: [
          { text: '计算机知识', link: '/IT-learning/408知识/'},
          { text: 'Java后端', link: '/IT-learning/Java/'},
          { text: 'Linux技术',  link: '/IT-learning/Linux/'},
          { text: '计算机图形学',  link: '/IT-learning/计算机图形学/'},
        ]
      },

      {
        text: '自我提升',
        items: [
          { text: '阅读', link: '/improve/阅读/' },
          { text: '金融投资', link: '/improve/金融投资/' },
        ]
      },
      {
        text: '求职面试',
        items: [
          { text: 'Java面经', link: '/面试求职/Java面经/' },
          { text: '场景问题', link: '/面试求职/场景问题/' },
          { text: '经验分享', link: '/面试求职/经验分享/' },
          { text: '算法岗', link: '/面试求职/算法岗/' },
        ],
      },
      {
        text: '其他维护',
        items: [
          { text: '站点更新', link: '/update/更新日志' },
          { text: '问题清单', link: '/技术问题清单/' },
        ]
      },
      {
        text: '感悟和日常',
        items: [
          { text: '站长感悟', link: '/我的感悟/' },
          { text: '站长旧版博客', link: 'https://EthanLiu6.github.io' },
        ]
      },
    ], // end导航栏

    /**
        // 导航栏跳转后的侧边栏
        sidebar: {
          '/guide/': [
            {
              text: '博客指南',
              collapsed: true,
              items: [
                { text: '博客md语法', link: '/guide/markdown-examples' },
                { text: 'API案例', link: '/guide/api-examples' },
                { text: 'API案例-cp', link: '/guide/api-examples copy' }
              ]
            }
          ],
          '/Java/': [
            {
              text: 'Java后端',
              collapsed: true,
              items: [
                { text: 'JavaSE', link: '/Java/java-se' },
                { text: 'JavaWeb', link: '/Java/java-web' },
                { text: 'SSM', link: '/Java/ssm' },
              ]
            }
          ],
          '/linux/': [
            {
              text: 'Linux技术',
              collapsed: true,
              items: [
                { text: 'Linux基础', link: '/linux/linux-basics' },
                { text: 'Shell脚本', link: '/linux/shell-scripts' },
                { text: '网络配置', link: '/linux/network-setup' },
              ]
            }
          ],
          '/computer/': [
            {
              text: '计算机知识',
              collapsed: true,
              items: [
                { text: '数据结构', link: '/computer/data-structures' },
                { text: '计算机网络', link: '/computer/computer-networks' },
                { text: '计算机组成原理', link: '/computer/computer-architecture' },
              ]
            }
          ],
          '/improve/': [
            {
              text: '个人提升',
              collapsed: true,
              items: [
                { text: '冥想练习', link: '/improve/meditation' },
                { text: '学科交叉', link: '/improve/interdisciplinary' },
                { text: '金融投资', link: '/improve/investment' },
              ]
            }
          ],
          '/interview/': [
            {
              text: '求职面试',
              collapsed: true,
              items: [
                { text: '面试技巧', link: '/interview/interview-tips' },
                { text: '常见问题', link: '/interview/common-questions' },
              ]
            }
          ],
          '/update/': [
            {
              text: '站点更新',
              collapsed: true,
              items: [
                { text: "更新日志", link: '/update/更新日志' },
                { text: "2024-10[建站]", link: '/update/2024-10[建站]' },
                { text: "2024-11", link: '/update/2024-11' },
              ],
            }
          ]
        }, // end侧边栏
     * 
     */

    sidebar: {
      
    },

    // 页脚配置
    footer: {
      message: 'ICP备案号: <a href="https://beian.miit.gov.cn/" target="_blank">蜀ICP备2024103116号</a><br>公安备案号: <a href="https://beian.mps.gov.cn/#/query/webSearch?code=51012202001928" rel="noreferrer" target="_blank">川公网安备51012202001928</a>',
      copyright: '版权所有 © 2024-present  <a href="mailto:16693226842@163.com" target="_blank">Ethan.Liu</a>'
    
    },

    // 编辑链接配置
    editLink: {
      pattern: 'https://github.com/EthanLiu6/coderethan.fun/edit/master/docs/:path',
      text: '在 GitHub 上编辑此页面 OR 提出修改意见'
    },

    // 最后更新时间配置
    lastUpdated: {
      text: '最后更新于', // 自定义最后更新时间的文本
      formatOptions: {
        dateStyle: 'long',
        timeStyle: 'short'
      }
    },

    docFooter: {
      prev: "上一篇",
      next: "下一篇",
    },

    // 深色模式切换
    darkModeSwitchLabel: '深色模式',
    lightModeSwitchTitle: '切换到浅色模式',
    darkModeSwitchTitle: '切换到深色模式'
  }
})
