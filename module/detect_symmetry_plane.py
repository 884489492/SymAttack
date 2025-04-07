import numpy as np
import torch
import open3d as o3d
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from PointNet import PointNetCls


def detect_SymElem(P, model, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                   tau_s=0.001, tau_P=64, tau_r=0.1, tau_sigma=0.01, alpha=0.5
                   ):
    """

    :param device:
    :param alpha: 对称调整步长
    :param tau_P: 部分级元素最大点数
    :param tau_sigma: 对称性验证阈值
    :param tau_r: 补丁级元素距离阈值
    :param tau_s: 语义距离阈值
    :param P: 干净的点云
    :param model:
    :return: 对称平面 S，对称元素集合
    """
    # 1、使用八叉树和主轴变换识别对称平面
    P_np = P if isinstance(P, np.ndarray) else P.cpu().numpy()

    octree = build_octree(P_np, max_depth=8, min_points=1)  # 构建一个八叉树来表示点云的空间划分

    pca = PCA(n_components=3)  # 用主成分分析来分析点云的分布方向实现主轴变换识别对称平面
    pca.fit(P_np)  # 计算点云的协方差矩阵，分解出特征值和特征向量
    center = np.mean(P_np, axis=0)
    normal = pca.components_[2]  # 把第三主轴作为对称平面法线
    # 平面方程是 ax + by + cz + d = 0, (a, b, c)是法向量 normal,
    # 通过点云中心 (x_c, y_c, z_c)的条件是 ax_c + by_c + cz_c + d = 0, 因此 d = -(ax_c + by_c + cz_c） = - normal·center
    d = -np.dot(normal, center)
    S = (normal, d)  # 将法向量 normal 和偏移量 d 组成元组，表示对称平面

    # 2、在对称平面一侧采样部分元素
    Q_idx = farthest_point(P_np, K=10)  # 些点在点云空间中分布均匀，能够代表点云的整体几何结构
    Q = P_np[Q_idx]
    P_tensor = torch.from_numpy(P_np).float().unsqueeze(0).to(device).transpose(1, 2)
    model.eval()
    with torch.no_grad():
        _, P_features = model(P_tensor)
    P_features = P_features.squeeze(0).detach().cpu().numpy()

    part_elements = []
    for q_idx in Q_idx:
        q_feature = P_features[:, q_idx]
        element = []
        for i in range(P_np.shape[0]):
            dist = np.linalg.norm(P_features[i] - q_feature)  # 语义距离
            if dist < tau_s:
                element.append(i)
        if 0 < len(element) <= tau_P:
            part_elements.append(np.array(element))

    # 采样补丁级元素
    occupied = np.concatenate(part_elements) if part_elements else np.array([])
    remaining = np.setdiff1d(np.arange(P_np.shape[0]), occupied)
    P_remain = P_np[remaining]
    patch_elements = []
    for q in Q:
        element = []  # element是索引数组
        for i in range(len(remaining)):
            if np.linalg.norm(P_remain[i] - q) < tau_r:
                element.append(remaining[i])
        if len(element) > 0:
            patch_elements.append(np.array(element))

    # 对称性验证,从部分级元素和补丁级元素的集合中识别出有对称性的元素对
    all_elements = part_elements + patch_elements
    symmetric_elements = []
    for E in all_elements:
        P_E = P_np[E]
        P_E_mirror = mirror_point(P_E, normal, d)
        mirrored_indices = []
        for p_m in P_E_mirror:  # 对每个镜像点检查它是否在点云中存在接近的对应点
            leaf_node, _ = octree.locate_leaf_node(p_m)
            if leaf_node:
                leaf_points = np.asarray(octree.root_node.get_points())  # 获取八叉树中的所有点
                if len(leaf_points) > 0:
                    closest_idx = np.argmin(np.linalg.norm(P_np - p_m, axis=1))   # 找到点云中距离镜像点的最近点的索引
                    mirrored_indices.append(closest_idx)
        if len(mirrored_indices) == len(E): # 检查点数匹配情况, 是否每个镜像点都能找到对应点
            avg_dist = np.mean( # 计算原始点和镜像点的平均距离
                [np.linalg.norm(P_np[i] - mirror_point(P_np[j], normal, d)) for i, j in zip(E, mirrored_indices)])
            if avg_dist < tau_sigma: # 验证对称性
                symmetric_elements.append({"original":E, "mirrored": np.array(mirrored_indices)})
    return S, symmetric_elements





def mirror_point(P, normal, d):  # 计算镜像点, 镜像点是原始点关于某个平面的反射, 从原始点到镜像点的向量垂直于平面,且长度是原始点到平面距离的2倍
    """

    :param P: 点云
    :param normal:法线
    :param d: 距离
    :return:
    对称平面方程是 n·x + d = = ax + by + cz + d = 0; n = [a, b, c], 是法向量; d 是平面到原点的距离
    点 P = [x_i, y_i, z_i] 到平面的有符号距离: dist = n · P + d = a·x_i + b·y_i + c·z_i + d; dist > 0表示点在法向量指向的一侧, <0 则在另一侧
    点 P 沿法线向量方向投影到平面的点 Q: Q = P - dist·n
    镜像点 P' 在 Q 的另一侧, 距离Q等于 dist, 从 Q 沿法向量反方向移动 dist: P' = Q - dist·n = P - dist·n - dist·n = P - 2*dist·n
    """
    dist = np.dot(P, normal) + d
    return P - 2 * dist[:, None] * normal


# 构建八叉树
def build_octree(P, max_depth=8, min_points=1):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(P)
    octree = o3d.geometry.Octree(max_depth)
    octree.convert_from_point_cloud(pcd, size_expand=0.01)
    return octree


def farthest_point(P, K):  # 采样最远点
    """
    目标: 从点云中选择 K 个点，使得这些点在空间中分布尽可能均匀以覆盖整个点云的几何结构
    通过逐步选择距离已有采样点最远的点来实现这一目标
    :param P: 点云数据
    :param K: 需要采样的点数
    :return: 包含 K 个采样点索引的数组
    """
    N = P.shape[0]
    selected = [0]  # 初始化采样点列表，选择第一个点作为起点
    distance = np.linalg.norm(P - P[0], axis=1)  # 计算点云中所有点到第一个采样点 P[0] 的欧几里得距离，表示每个点到当前采样点集的最小距离
    for _ in range(K - 1):
        """
        选择最远点: 目标: 从点云 P 中选择 K 个点, 这些点要尽可能均匀分布, 覆盖整个点云空间
        每次选择距离当前采样点集最远的点，逐步扩展采样点集
        找到distance数组中最大值的索引,也就是距离采样点集最远的点，这样意味着它在空间中还没有被重复覆盖，选择这个点加入selected可以扩展采样点集的覆盖范围，增加均匀性
        """
        idx = np.argmax(distance)  # distance表示示每个点到采样点集的最近距离，idx是下一步要选择的点，必须是当前最远的点，
        selected.append(idx)
        # 最近覆盖距离
        """
        distance的值反映了点云被采样点集覆盖的程度,值越小表示越靠近某个采样点，更新后的distance用于下一次np.argmax以便找到新的最远点
        """
        distance = np.minimum(distance, np.linalg.norm(P - P[idx], axis=1))  # 更新每个点到采样点集的最小距离,确保每个点到采样点集的最近距离是最新的
    return np.array(selected)
