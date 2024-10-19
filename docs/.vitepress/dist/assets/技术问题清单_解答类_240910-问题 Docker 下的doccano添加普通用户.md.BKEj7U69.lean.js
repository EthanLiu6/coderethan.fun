import{_ as a,c as i,a2 as n,o as e}from"./chunks/framework.C9NVOr0y.js";const d=JSON.parse('{"title":"","description":"","frontmatter":{},"headers":[],"relativePath":"技术问题清单/解答类/240910-问题 Docker 下的doccano添加普通用户.md","filePath":"技术问题清单/解答类/240910-问题 Docker 下的doccano添加普通用户.md","lastUpdated":1729103515000}'),p={name:"技术问题清单/解答类/240910-问题 Docker 下的doccano添加普通用户.md"};function l(t,s,h,c,k,o){return e(),i("div",null,s[0]||(s[0]=[n(`<h2 id="_240910-问题-docker下的doccano添加普通用户" tabindex="-1">240910-问题 Docker下的doccano添加普通用户 <a class="header-anchor" href="#_240910-问题-docker下的doccano添加普通用户" aria-label="Permalink to &quot;240910-问题 Docker下的doccano添加普通用户&quot;">​</a></h2><div class="tip custom-block"><p class="custom-block-title">整体实现方式</p><p>通过不断排查发现，24年最新版本的有一些bug，无法创建普通用户，只能创建超管然后将别的超管作为员工添加到项目中。</p></div><h3 id="_1-操作步骤" tabindex="-1">1. 操作步骤 <a class="header-anchor" href="#_1-操作步骤" aria-label="Permalink to &quot;1. 操作步骤&quot;">​</a></h3><h3 id="_1-1-启动docker的doccano" tabindex="-1">1.1 启动docker的doccano <a class="header-anchor" href="#_1-1-启动docker的doccano" aria-label="Permalink to &quot;1.1 启动docker的doccano&quot;">​</a></h3><div class="language-bash vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">bash</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"># 需要提前创建好doccano服务才行（docker run …………）</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">docker</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> restart</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> doccano</span></span></code></pre></div><h3 id="_1-2-进入doccano容器" tabindex="-1">1.2 进入Doccano容器 <a class="header-anchor" href="#_1-2-进入doccano容器" aria-label="Permalink to &quot;1.2 进入Doccano容器&quot;">​</a></h3><div class="language-sh vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">sh</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">docker</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> exec</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> -it</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> doccano</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> bash</span></span></code></pre></div><h3 id="_1-3-配置-添加-超级管理员或者普通管理员账号" tabindex="-1">1.3 配置（添加）超级管理员或者普通管理员账号 <a class="header-anchor" href="#_1-3-配置-添加-超级管理员或者普通管理员账号" aria-label="Permalink to &quot;1.3 配置（添加）超级管理员或者普通管理员账号&quot;">​</a></h3><blockquote><p>因为doccano后端是采用Django框架搭建的，可以直接使用 manage.py 进行创建账户</p></blockquote><h4 id="_1-3-1-管理员账户" tabindex="-1">1.3.1 <strong>管理员账户</strong> <a class="header-anchor" href="#_1-3-1-管理员账户" aria-label="Permalink to &quot;1.3.1 **管理员账户**&quot;">​</a></h4><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span># python manage.py createsuperuser --username &lt;用户名&gt; --email &lt;邮箱&gt; --password &lt;密码&gt;</span></span>
<span class="line"><span>python manage.py createsuperuser --username demo  --password wazg123456</span></span></code></pre></div><p>这里我用我的远程win系统给大家演示</p><img src="https://coderethan-1327000741.cos.ap-chengdu.myqcloud.com/blog-pics/image-20241010220222677.png" alt="image-20241010220222677" style="zoom:67%;"><p>远程机访问测试管理员账户</p><img src="https://coderethan-1327000741.cos.ap-chengdu.myqcloud.com/blog-pics/image-20241010220329631.png" alt="image-20241010220329631" style="zoom:50%;"><p>登录成功：</p><img src="https://coderethan-1327000741.cos.ap-chengdu.myqcloud.com/blog-pics/image-20241010220352067.png" alt="image-20241010220352067" style="zoom:50%;"><h4 id="_1-3-2-普通用户账户" tabindex="-1">1.3.2 <strong>普通用户账户</strong> <a class="header-anchor" href="#_1-3-2-普通用户账户" aria-label="Permalink to &quot;1.3.2 **普通用户账户**&quot;">​</a></h4><p>我使用createuser 发现没了，然后查看manage.py help</p><p><img src="https://coderethan-1327000741.cos.ap-chengdu.myqcloud.com/blog-pics/image-20241010224949885.png" alt="image-20241010224949885"></p><p>结果发现只有这些api</p><div class="language-shell vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">shell</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">Available</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> subcommands:</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">[account]</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">    account_unsetmultipleprimaryemails</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">[api]</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">    create_admin</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">    wait_for_db</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">[auth]</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">    changepassword</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">    createsuperuser</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">[authtoken]</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">    drf_create_token</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">[contenttypes]</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">    remove_stale_contenttypes</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">[django]</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">    check</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">    compilemessages</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">    createcachetable</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">    dbshell</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">    diffsettings</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">    dumpdata</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">    flush</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">    inspectdb</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">    loaddata</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">    makemessages</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">    makemigrations</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">    migrate</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">    optimizemigration</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">    sendtestemail</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">    shell</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">    showmigrations</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">    sqlflush</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">    sqlmigrate</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">    sqlsequencereset</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">    squashmigrations</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">    startapp</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">    startproject</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">    test</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">    testserver</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">[drf_yasg]</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">    generate_swagger</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">[health_check]</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">    health_check</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">[projects]</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">    create_member</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">[rest_framework]</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">    generateschema</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">[roles]</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">    create_roles</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">[runserver_nostatic]</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">    runserver</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">[sessions]</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">    clearsessions</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">[staticfiles]</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">    collectstatic</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">    findstatic</span></span></code></pre></div><p>于是我尝试使用<code>create_member</code>的api，结果发现会有错误</p><p><img src="https://coderethan-1327000741.cos.ap-chengdu.myqcloud.com/blog-pics/image-20241010225307601.png" alt="image-20241010225307601"></p><p>于是排查错误，<code>cat</code>查看<code>create_member.py</code>内容</p><p><img src="https://coderethan-1327000741.cos.ap-chengdu.myqcloud.com/blog-pics/image-20241010225745370.png" alt="image-20241010225745370"></p><p>尝试修改代码，但是，由于需要安装vim（或者其他编辑器），而且需要管理员权限，最终我放弃了这种策略</p><p>由于前面我看的manage.py的api还有一个<code>create_admin</code>，我尝试创建，最终效果与<code>createsuperuser</code>一模一样，只是create_admin需要添加名称和密码，createsuperuser的使用前面讲过了，这里展示create_admin</p><div class="language-python vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">python</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">#python manage.py create_admin --username &lt;成员名&gt; --password &lt;成员密码&gt;</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">python manage.py create_admin </span><span style="--shiki-light:#B31D28;--shiki-light-font-style:italic;--shiki-dark:#FDAEB7;--shiki-dark-font-style:italic;">--</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">username demo </span><span style="--shiki-light:#B31D28;--shiki-light-font-style:italic;--shiki-dark:#FDAEB7;--shiki-dark-font-style:italic;">--</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">password wazg123456</span></span></code></pre></div><p><img src="https://coderethan-1327000741.cos.ap-chengdu.myqcloud.com/blog-pics/image-20241010231213507.png" alt="image-20241010231213507"></p><p>登录后发现创建成功了，这里就不展示了</p><p>最后我们直接用管理项目的账号去登录，然后添加我们创建的这些管理员（成员）即可</p><p>依次点开： 创建的项目——&gt;侧边栏——&gt;成员（member）——&gt;增加（add）——&gt;选择成员列表并设置权限——&gt;保存</p><p><img src="https://coderethan-1327000741.cos.ap-chengdu.myqcloud.com/blog-pics/image-20241010231720882.png" alt="image-20241010231720882"></p><p><img src="https://coderethan-1327000741.cos.ap-chengdu.myqcloud.com/blog-pics/image-20241010231817768.png" alt="image-20241010231817768"></p><p><img src="https://coderethan-1327000741.cos.ap-chengdu.myqcloud.com/blog-pics/image-20241010231844814.png" alt="image-20241010231844814"></p>`,36)]))}const g=a(p,[["render",l]]);export{d as __pageData,g as default};
