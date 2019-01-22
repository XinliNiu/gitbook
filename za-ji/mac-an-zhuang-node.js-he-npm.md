---
description: 在mac上安装node.js和npm
---

# Mac安装Node.js和npm

首先访问node.js官网（https://nodejs.org/en/download/）  
![](https://images2015.cnblogs.com/blog/998610/201611/998610-20161110142509592-1402406884.png)

点击下载完后，一路点安装 就安装完成了

然后打开-终端-输入node -v 会返回当前安装的版本号 看图  
![](https://images2015.cnblogs.com/blog/998610/201611/998610-20161110142725249-900055982.png)

然后在输入 npm -v 得到  
![](https://images2015.cnblogs.com/blog/998610/201611/998610-20161110142757749-821427894.png)  
这算安装正常了   


写一个测试代码helloworld.js

```javascript
const http = require('http');
const hostname = '127.0.0.1';
const port = 1337;
http.createServer((req, res) => {
res.writeHead(200, { 'Content-Type': 'text/plain' });
res.end('Hello World\n');
}).listen(port, hostname, () => {
console.log(`Server running at http://${hostname}:${port}/`);
});
```

执行

![](https://images2015.cnblogs.com/blog/998610/201611/998610-20161110143209014-2007327076.png)

然后把下面的ip地址复制到浏览器打开

![](https://images2015.cnblogs.com/blog/998610/201611/998610-20161110143313233-1151696868.png)

_这就配置成功了_

