# 内存回收



## Hotspot内存回收概述

### 分代垃圾回收\(Generational Garbage Collection\)

Hotspot JVM使用的是分代垃圾回收，它主要基于以下两个观察：

* 大部分对象很快就变得不可达了
* 很少有老的对象引用新的对象的情况

举个例子，我们有个交易，每次去数据库里查询一条数据返回。

```java
Map response = queryInfo(); //从数据库里查出数据并新建一个HashMap放到response里
printWriter.write(response);
```

这个交易被大量的调用，但是交易中new的对象存在的时间很短。这个就是大部分对象很快就变得不可达，可以回收了。

而老的对象，例如长时间存在的配置信息，产品信息等对象。我们一般不会去动它，不会经常的修改它导致它引用了新产生的对象，所以老对象引用的对象也一般是老的。

上面两个假设称为“弱年代假设“， hotspot的gc就是基于这个假设来做的，它把堆分成了两个物理区域，新生代\(young generation\)和老年代\(old generation\)。

* 新生代：大部分新分配的对象都会放到新生代，新生代比较小，而且被回收的频率很高，对它的回收称为young gc或者minor gc。因为大部分对象的生存周期很短，比如90％的新对象都在一个young gc周期内实效，那么我们的young gc能回收大部分的对象，而且比较快，因为它只扫描新生代。
* 老年代：在young gc的时候，可能有些对象回收不掉，逃过若干次young gc的对象，会被升级\(promote或者tenure\)到老年代，因为这些对象是有很大几率长期存在到，如果一直放到新生代，会占据新生代内存，还会影响young gc到效率，所以把它们挪到老年代让他们养老。对老年代对的回收称为old gc或者major gc或者full gc，因为老年代生成对象的速度较慢，有些应用都基本不增长，所以gc很少。
* 永久代：这个代是存储元数据的，例如类信息，一般不参与gc。

（图：promote）

young gc需要做到效率很高，回收器需要高效的找到新生代中存活的对象就不能扫描整个老年代。在hotspot里用到了一种数据结构叫card table。 老年代被分成了512B大小的块，称为card。card table是一个堆上的数组，数组里存的是byte。每次更新一个对象里的引用字段的时候，必须更新该应用字段对应的card，标记为dirty，这样在young gc的时候只需要扫描card table。

在调用更新对象里的引用字段的时候，都需要执行一个写屏障\(write barrier\)，这样会引入一些性能开销，但是用它来高效的实现young gc是很值得的。（需要深究）

### 新生代\(The young generation\)

新生代又分成了3个区域：eden space, survivor space\(from, to\)。

* The eden：大部分的新对象都是直接分配在eden space里的（不全是，因为有些大对象直接在老年代分配）。eden区在通常情况下经过一次young gc后就会被清空（有个例不为空）。
* The two survivor spaces: 这里存的是至少逃过一次young gc的对象，但是还不足以进入老年代。对于这两个区，from和to，只有一个被使用，另一个是空的。它们两个交替切换为from和to。

young gc回收步骤如下：

1. 找到eden和from区里存活的对象
2. eden区里存活对象全部复制到to区中
3. from区中的存活对象，还不够养老资格的，继续复制到to区中
4. from区中的存活对象，已经满足了young gc的次数，复制到老年代
5. 清空eden区，把from变成to，to变成from

因为回收过程中要复制内存，这种回收器被称为基于复制的垃圾回收。

以上存在一个问题：在回收的过程中，from区和eden区的存活对象都要放到to区，如果to区空间不够，怎么办？

在移动过程中，发现to区不够了，那么剩下的对象需要放到老年代里，这被称为“早熟提升“\(premature promotion\)。这样老年代就会存生命周期短的对象，如果满了，就会进行full gc，这被称为“提升失败“\(promotion failure\)。只要优化得当，这两种情况一般比较少出现。

### 快速分配\(Fast Allocation\)

垃圾回收与对象分配紧密相关，因为所有的对象分配都在eden区，所以分配的时候直接顺序分配就行了，用了bump-the-pointer技巧，就是维护一个已分配内存的结束指针，每次分配的时候看看剩下的内存够不够用，如果够用就移动这个结束指针，否则就得内存回收了。

但是因为JVM是多线程的，为了保证线程安全，需要使用全局锁，这样会影响性能。为了改进这个不足，HotSpot把eden区分成了多个线程局部分配缓存\(Thread-Local Allocation Buffers, TLABs\)。每个线程申请内存时，如果自己没有可用的TLAB了，就给分配一个新的，如果有可用的，那就直接在自己当前可用的TLAB上分配，不需要上锁，但是新申请一个TLAB需要上锁。

在HotSpot里，new Object\(\)只用了大约10条汇编指令，这个开销是非常小的，就是得益于eden区每次都会被清空。

## 内存回收算法

Hotspot里主要有四种回收算法，Serial GC, Parallel GC, CMS GC, G1 GC。

### Serial GC

Serial GC里对young gc的操作就如前边所述，young gc的时候需要stop the world。对老年代对回收是通过"滑动压缩标记清除"\(sliding compacting mark-sweep\)，也被称为标记压缩垃圾回收器\(mark-compact\)，它也是需要stop the world。

mark-compact首先标记出老年代里的存活的对象，然后把它们移动到老年代的开始部分







