import torch
import torch.nn as nn
import torch.nn.functional as F


def sym_attack(point_cloud, classifier, label, symElm, symPlane, max_iter=100, epsilon=0.01, alpha=0.5):
    """

    :param point_cloud: 原始点云
    :param classifier: 点云分类器, 用来计算损失
    :param label:  真实标签
    :param symElm:  包含原始点索引和镜像点索引的对称元素对列表, 每个元素是个字典
    :param symPlane: 对称平面, 一个四元组 (a, b, c, d)，表示平面 ax+by+cz+d=0
    :param max_iter: 最大迭代次数
    :param epsilon: 扰动步长
    :param alpha: 对称调整步长
    :return: adv_cloud, 对抗点云, 在原始点云上加了扰动后依然保持着对称性

    """
    adv_cloud = point_cloud.clone().requires_grad_(True)

    # 生成初始扰动
    for iteration in range(max_iter):
        classifier.zero_grad()
        classifier.eval()
        input_tensor = adv_cloud.unsqueeze(0).transpose(1, 2)
        output = classifier(input_tensor)
        loss = -F.cross_entropy(output, label)    # 误分类, 负号表示最大损失
        loss.backward()
        grad = adv_cloud.grad.data    # 获取梯度, 表示每个点在(x, y, z)方向的扰动方向

        # 更新扰动
        with torch.no_grad():
            adv_cloud += epsilon * grad.sign()
            adv_cloud.grad.zero_()

    # 对称感知调整
    """
    方向调整的目标: 保证对称点对(原始点 P, 和镜像点 P_m)的扰动向量保持对称性, 点 P 的扰动向量 d_p和 P_m 的扰动向量 d_p_m互为镜像
    """
    for elem in symElm:
        original_indices = elem["original"]
        mirrored_indices = elem["mirrored"]
        for p_idx, p_mirror_idx in zip(original_indices, mirrored_indices):
            # d_p = P' - P，表示 P -> P' 的扰动
            d_p = adv_cloud[p_idx] - point_cloud[p_idx]  # 计算原始点 P 的扰动向量用来表示当前对抗点 P' 相对于原始点 P的偏移

            # d_p_m = P_m' - P_m，表示 P_m -> P_m' 的扰动
            d_p_mirror = adv_cloud[p_mirror_idx] - point_cloud[p_mirror_idx]  # 计算镜像点 P_m 的扰动向量

            # 方向调整 如果 P 和 P_m 是对称的，那么扰动向量也应该是对称的
            # 扰动向量 d_p_m的镜像, 是 d_p的目标
            mirrored_d_p_mirror = mirror_vector(d_p_mirror, symPlane)  # 将镜像点的扰动向量关于对称平面反射来得到它的镜像向量

            # 扰动向量 d_p的镜像，是 d_p_m的目标
            mirrored_d_p = mirror_vector(d_p, symPlane)   # 将原始点的扰动向量反射
            # 调整 d_p, 向 d_p_m的镜像看齐
            new_d_p = d_p + alpha * (mirrored_d_p_mirror - d_p)  # 调整 d_p让他趋向于d_m的镜像方向
            # 调整 d_p_m，向 d_p的镜像看齐
            new_d_p_mirror = d_p_mirror + alpha * (mirrored_d_p - d_p_mirror)
            new_d_p = F.normalize(new_d_p, dim=0)
            new_d_p_mirror = F.normalize(new_d_p_mirror, dim=0)

            # 调整幅度
            # 计算d_p和 d_p_m的模长, 代表着他们各自的扰动大小
            sigma_p = torch.norm(d_p)
            sigma_p_mirror = torch.norm(d_p_mirror)
            # 取原始点和镜像点扰动幅度的较小值,对称点的扰动幅度应该一致同时为了保持不可感知性也不能过大, 选择较小的幅度，确保扰动不会因某一方过大而破坏整体对称性或显著改变点云外观
            sigma = min(sigma_p, sigma_p_mirror)

            # 更新原始点云
            adv_cloud[p_idx] = point_cloud[p_idx] + sigma * new_d_p
            # 更新镜像点云
            adv_cloud[p_mirror_idx] = point_cloud[p_mirror_idx] + sigma * new_d_p_mirror
        adv_cloud = adv_cloud.detach().requires_grad(True)
    return adv_cloud.detach()



def mirror_vector(vector, symPlane):    # 镜像向量
    a, b, c, d = symPlane
    normal = torch.tensor([a, b, c], dtype=torch.float32)
    normal = normal / torch.norm(normal)    
    dot_product = torch.dot(vector, normal)
    mirrored_vector = vector - 2 * dot_product * normal
    return mirrored_vector

