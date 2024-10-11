### 240910-问题 Docker下的doccano添加普通用户

::: tip 整体实现方式
通过不断排查发现，24年最新版本的有一些bug，无法创建普通用户，只能创建超管然后将别的超管作为员工添加到项目中。
:::

#### 1. 操作步骤

#### 1.1 启动docker的doccano

```bash
# 需要提前创建好doccano服务才行（docker run …………）
docker restart doccano
```

#### 1.2 进入Doccano容器

```sh
docker exec -it doccano bash
```

#### 1.3 配置（添加）超级管理员或者普通管理员账号

> 因为doccano后端是采用Django框架搭建的，可以直接使用 manage.py 进行创建账户

##### 1.3.1 **管理员账户**

```
# python manage.py createuser --username <用户名> --email <邮箱> --password <密码>
python manage.py createuser --username demo  --password wazg123456
```

这里我用我的远程win系统给大家演示

<img src="https://coderethan-1327000741.cos.ap-chengdu.myqcloud.com/blog-pics/image-20241010220222677.png" alt="image-20241010220222677" style="zoom:67%;" />

远程机访问测试管理员账户

<img src="https://coderethan-1327000741.cos.ap-chengdu.myqcloud.com/blog-pics/image-20241010220329631.png" alt="image-20241010220329631" style="zoom: 50%;" />



登录成功：

<img src="https://coderethan-1327000741.cos.ap-chengdu.myqcloud.com/blog-pics/image-20241010220352067.png" alt="image-20241010220352067" style="zoom:50%;" />

##### 1.3.2 **普通用户账户**

我使用createuser 发现没了，然后查看manage.py help

![image-20241010224949885](https://coderethan-1327000741.cos.ap-chengdu.myqcloud.com/blog-pics/image-20241010224949885.png)

结果发现只有这些api

```shell
Available subcommands:

[account]
    account_unsetmultipleprimaryemails

[api]
    create_admin
    wait_for_db

[auth]
    changepassword
    createsuperuser

[authtoken]
    drf_create_token

[contenttypes]
    remove_stale_contenttypes

[django]
    check
    compilemessages
    createcachetable
    dbshell
    diffsettings
    dumpdata
    flush
    inspectdb
    loaddata
    makemessages
    makemigrations
    migrate
    optimizemigration
    sendtestemail
    shell
    showmigrations
    sqlflush
    sqlmigrate
    sqlsequencereset
    squashmigrations
    startapp
    startproject
    test
    testserver

[drf_yasg]
    generate_swagger

[health_check]
    health_check

[projects]
    create_member

[rest_framework]
    generateschema

[roles]
    create_roles

[runserver_nostatic]
    runserver

[sessions]
    clearsessions

[staticfiles]
    collectstatic
    findstatic
```

于是我尝试使用`create_member`的api，结果发现会有错误

![image-20241010225307601](https://coderethan-1327000741.cos.ap-chengdu.myqcloud.com/blog-pics/image-20241010225307601.png)

于是排查错误，`cat`查看`create_member.py`内容

![image-20241010225745370](https://coderethan-1327000741.cos.ap-chengdu.myqcloud.com/blog-pics/image-20241010225745370.png)

尝试修改代码，但是，由于需要安装vim（或者其他编辑器），而且需要管理员权限，最终我放弃了这种策略



由于前面我看的manage.py的api还有一个`create_admin`，我尝试创建，最终效果与`createsuperuser`一模一样，只是create_admin需要添加名称和密码，createsuperuser的使用前面讲过了，这里展示create_admin

![image-20241010231213507](https://coderethan-1327000741.cos.ap-chengdu.myqcloud.com/blog-pics/image-20241010231213507.png)

登录后发现创建成功了，这里就不展示了

最后我们直接用管理项目的账号去登录，然后添加我们创建的这些管理员（成员）即可

依次点开：  创建的项目——>侧边栏——>成员（member）——>增加（add）——>选择成员列表并设置权限——>保存

![image-20241010231720882](https://coderethan-1327000741.cos.ap-chengdu.myqcloud.com/blog-pics/image-20241010231720882.png)

![image-20241010231817768](https://coderethan-1327000741.cos.ap-chengdu.myqcloud.com/blog-pics/image-20241010231817768.png)

![image-20241010231844814](https://coderethan-1327000741.cos.ap-chengdu.myqcloud.com/blog-pics/image-20241010231844814.png)
