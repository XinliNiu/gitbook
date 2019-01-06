---
description: >-
  Kubernetes是一个可移植可扩展的开源容器和服务管理平台，既支持声明式配置，也支持自动化配置。
  它的生态目前已经发展的很大并且在快速增长，kubernetes的服务、支持和工具随处可见。谷歌在2014年开源了Kubernetes，它是基于谷歌15年在生产上大规模使用容器的经验，结合了社区里优秀的想法和实践。
---

# 概念

## 为什么我需要Kubernetes以及它能做什么？

k8s有很多特性，它可以被看成：

* 一个容器平台
* 一个微服务平台
* 一个可移植的云平台，以及更多。。

k8s提供了以容器为中心\(container-centric\)的管理环境。它根据用户的负载\(user load\)调整\(orchestrate\)计算资源、网络和存储等基础设施。它既提供了PaaS的简单易用特性，又提供了IaaS的灵活性，并且可以在不同的基础设施提供商之间移植。

## 为什么Kubernetes是一个平台？

尽管Kubernetes提供了很多功能，但是仍会不断的有新场景从新功能中受益。应用特定的工作流可以被流式处理，加快了开发速度。天然支持资源临时分配，得益于强大的自动伸缩功能。所以k8s被设计成一个组件和工具的生态系统，让部署、扩容和管理应用更加简单。

Labels empower users to organize their resources however they please. Annotations enable users to decorate resources with custom information to facilitate their workflows and provide an easy way for management tools to checkpoint state.

Additionally, the Kubernetes control plane is built upon the same APIs that are available to developers and users. Users can write their own controllers, such as schedulers, with their own APIs that can be targeted by a general-purpose command-line tool.

This design has enabled a number of other systems to build atop Kubernetes.

## What Kubernetes is not?

K8s is not a traditional, all-inclusive PaaS\(Platform as a Service\) system. Since kubernetes operates at the container level rather than the hardware level, it provides some generally applicable features common to PaaS offerings, such as deployment, scaling, loadbalancing, logging, and monitoring. However, Kubernetes is not monolithic, and these default solutions are optional and pluggable. Kubernetes provides the building blocks for building developer platforms, but preserves user choice and flexibility where it is important.

Kubernetes:

* Does not limit the types of applications supported. Kubernetes aims to support an extremely diverse variety of workloads, including stateless, stateful, and data-processing workloads. If an application can run in a container, it should run great on Kubernetes.
* Does not deploy source code and does not build your application. Continuous Integration, Delivery, and Deployment\(CI/CD\) workflows are determined by organization cultures and preferences as well as technical requirements.
* Does not provide application-level services, such as middleware\(e.g., message buses\), data-processing frameworks\(for example, Spark\), databases\(e.g., mysql\), caches, nor cluster storage system\(e.g., Ceph\) as built-in services. Such components can run on Kubernetes, and/or can be accessed by applications running on Kubernetes through portable mechanisms, such as the Open Service Broker.
* Does not dictate logging, monitoring, or alerting solutions. It provides some integrations as proof of concept, and mechanisms to collect and export metrics.
* Does not provide nor mandate a configuration language/system\(e.g., jsonnet\). It provides a declarative API that may be targeted by arbitrary forms of declarative specifications.
* Does not provide nor adopt any comprehensive machine configuration, maintenance, management, or self-healing systems.

Additionally, Kubernetes is not a mere orchestration system. In fact, it eliminates the need for orchestration. The technical definition of orchestration is execution of a defined workflow: first do A, then B, then C. In contrast, Kubernetes is comprised of a set of independent, composable control processes that continuously drive the current state towards the provided desired state. It shouldn't matter how you get from A to C. Centralized control is also not required. This results in a system that is easier to use and more powerful, robust, resilient , and extensible.

Why containers?



