import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import mahalanobis
import matplotlib.pyplot as plt
from sklearn.metrics import davies_bouldin_score
import redis
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class TimeSeriesClusteringPlatform:
    """
    基于G-G聚类算法的时间序列聚类平台
    用于室内设计客户管理平台的负载压力分段
    """
    
    def __init__(self, n_clusters_range=(2, 10)):
        """
        初始化时间序列聚类器
        
        参数:
        n_clusters_range: 聚类数量范围
        """
        self.n_clusters_range = n_clusters_range
        self.optimal_n_clusters = None
        self.cluster_centers_ = None
        self.labels_ = None
        self.mdbis = []
        
    def load_data(self, data_path):
        """
        加载时间序列数据
        假设数据格式为CSV，包含时间戳和访问次数
        """
        df = pd.read_csv(data_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        return df
    
    def preprocess_data(self, df):
        """
        数据预处理：降维和标准化
        使用KICA（这里简化为PCA）进行降维
        """
        # 提取时间特征
        df['hour'] = df.index.hour
        df['minute'] = df.index.minute
        df['weekday'] = df.index.weekday
        
        # 使用PCA降维
        pca = PCA(n_components=3)
        features = df[['hour', 'minute', 'weekday', 'visit_count']].values
        features_reduced = pca.fit_transform(features)
        
        return features_reduced
    
    def calculate_mdbi(self, X, labels):
        """
        计算MDBI指数（Modified Davies-Bouldin Index）
        """
        try:
            # 使用Davies-Bouldin指数
            db_score = davies_bouldin_score(X, labels)
            # 转换为MDBI（这里假设MDBI是DBI的改进版本，简化处理）
            mdbi = 1 / (1 + db_score) if db_score != 0 else 0
            return mdbi
        except:
            return 0
    
    def g_g_clustering(self, X, n_clusters, max_iter=100):
        """
        G-G聚类算法实现
        基于高斯混合模型的模糊聚类
        """
        # 使用高斯混合模型进行聚类
        gmm = GaussianMixture(n_components=n_clusters, 
                              covariance_type='full',
                              max_iter=max_iter,
                              random_state=42)
        gmm.fit(X)
        
        # 获取聚类标签和概率
        labels = gmm.predict(X)
        probabilities = gmm.predict_proba(X)
        
        # 计算聚类中心
        centers = gmm.means_
        
        return labels, probabilities, centers, gmm
    
    def find_optimal_clusters(self, X):
        """
        寻找最优聚类数量
        """
        best_mdbi = -np.inf
        best_n = 2
        
        for n in range(self.n_clusters_range[0], self.n_clusters_range[1] + 1):
            try:
                # 执行G-G聚类
                labels, probabilities, centers, gmm = self.g_g_clustering(X, n)
                
                # 计算MDBI
                mdbi = self.calculate_mdbi(X, labels)
                self.mdbis.append((n, mdbi))
                
                print(f"聚类数量: {n}, MDBI值: {mdbi:.4f}")
                
                # 更新最优聚类数量
                if mdbi > best_mdbi:
                    best_mdbi = mdbi
                    best_n = n
                    self.gmm = gmm
                    self.cluster_centers_ = centers
                    
            except Exception as e:
                print(f"聚类数量 {n} 失败: {e}")
                continue
        
        self.optimal_n_clusters = best_n
        print(f"\n最优聚类数量: {best_n}, 最优MDBI值: {best_mdbi:.4f}")
        return best_n
    
    def fuzzy_segmentation(self, X, timestamps):
        """
        时间序列模糊分段
        """
        if not hasattr(self, 'gmm'):
            self.find_optimal_clusters(X)
        
        # 获取每个时间点属于每个聚类的概率
        probabilities = self.gmm.predict_proba(X)
        
        # 创建分段结果
        segments = []
        current_segment = []
        current_cluster = None
        
        for i, (prob, ts) in enumerate(zip(probabilities, timestamps)):
            cluster = np.argmax(prob)
            
            if current_cluster is None:
                current_cluster = cluster
                current_segment.append((ts, cluster, prob[cluster]))
            elif cluster == current_cluster:
                current_segment.append((ts, cluster, prob[cluster]))
            else:
                # 开始新分段
                segments.append({
                    'start_time': current_segment[0][0],
                    'end_time': current_segment[-1][0],
                    'cluster': current_cluster,
                    'avg_probability': np.mean([p[2] for p in current_segment]),
                    'data_points': len(current_segment)
                })
                current_segment = [(ts, cluster, prob[cluster])]
                current_cluster = cluster
        
        # 添加最后一个分段
        if current_segment:
            segments.append({
                'start_time': current_segment[0][0],
                'end_time': current_segment[-1][0],
                'cluster': current_cluster,
                'avg_probability': np.mean([p[2] for p in current_segment]),
                'data_points': len(current_segment)
            })
        
        return segments
    
    def plot_clustering_results(self, X, timestamps, segments):
        """
        可视化聚类结果
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 原始时间序列
        axes[0, 0].plot(timestamps, X[:, 0], 'b-', alpha=0.7, label='访问量')
        axes[0, 0].set_title('原始访问量时间序列')
        axes[0, 0].set_xlabel('时间')
        axes[0, 0].set_ylabel('标准化访问量')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 聚类结果
        labels = self.gmm.predict(X)
        colors = plt.cm.Set3(np.linspace(0, 1, self.optimal_n_clusters))
        
        for cluster_id in range(self.optimal_n_clusters):
            mask = labels == cluster_id
            axes[0, 1].scatter(timestamps[mask], X[mask, 0], 
                              c=[colors[cluster_id]], 
                              label=f'Cluster {cluster_id}', 
                              alpha=0.6, s=50)
        
        axes[0, 1].set_title('时间序列聚类结果')
        axes[0, 1].set_xlabel('时间')
        axes[0, 1].set_ylabel('标准化访问量')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. MDBI值随聚类数量变化
        clusters_n = [m[0] for m in self.mdbis]
        mdbi_values = [m[1] for m in self.mdbis]
        
        axes[1, 0].plot(clusters_n, mdbi_values, 'ro-', linewidth=2, markersize=8)
        axes[1, 0].axvline(x=self.optimal_n_clusters, color='g', linestyle='--', 
                          label=f'最优聚类数: {self.optimal_n_clusters}')
        axes[1, 0].set_title('MDBI指数随聚类数量变化')
        axes[1, 0].set_xlabel('聚类数量')
        axes[1, 0].set_ylabel('MDBI值')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 时间分段可视化
        for seg in segments:
            color = colors[seg['cluster']]
            axes[1, 1].axvspan(seg['start_time'], seg['end_time'], 
                              alpha=0.3, color=color,
                              label=f'Cluster {seg["cluster"]}' if seg['cluster'] not in 
                              [s['cluster'] for s in segments[:segments.index(seg)]] else "")
        
        axes[1, 1].plot(timestamps, X[:, 0], 'b-', alpha=0.7)
        axes[1, 1].set_title('时间序列模糊分段结果')
        axes[1, 1].set_xlabel('时间')
        axes[1, 1].set_ylabel('标准化访问量')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # 打印分段结果
        print("\n时间序列分段结果:")
        print("-" * 80)
        for i, seg in enumerate(segments):
            print(f"分段 {i+1}:")
            print(f"  开始时间: {seg['start_time']}")
            print(f"  结束时间: {seg['end_time']}")
            print(f"  所属聚类: {seg['cluster']}")
            print(f"  平均概率: {seg['avg_probability']:.4f}")
            print(f"  数据点数: {seg['data_points']}")
            print("-" * 80)


class DistributedPlatform:
    """
    分布式室内设计客户管理平台模拟
    """
    
    def __init__(self):
        self.redis_client = None
        self.nodes = []
        self.load_history = []
        
    def init_redis(self, host='localhost', port=6379, db=0):
        """初始化Redis缓存"""
        try:
            self.redis_client = redis.Redis(host=host, port=port, db=db)
            print("Redis缓存初始化成功")
        except:
            print("警告: Redis连接失败，使用内存缓存模拟")
            self.redis_client = None
            self.cache = {}
    
    def get_from_cache(self, key):
        """从缓存获取数据"""
        if self.redis_client:
            try:
                data = self.redis_client.get(key)
                if data:
                    return json.loads(data)
            except:
                return None
        else:
            return self.cache.get(key)
    
    def set_to_cache(self, key, value, expire=3600):
        """设置缓存数据"""
        if self.redis_client:
            try:
                self.redis_client.setex(key, expire, json.dumps(value))
            except:
                pass
        else:
            self.cache[key] = value
    
    def dynamic_scaling(self, current_load, segments, threshold_high=0.8, threshold_low=0.3):
        """
        动态扩缩容
        根据负载压力调整节点数量
        """
        recommendations = []
        
        for seg in segments:
            if seg['avg_probability'] > threshold_high:
                # 高压时段，建议增加节点
                recommendations.append({
                    'time_period': f"{seg['start_time']} - {seg['end_time']}",
                    'recommendation': '增加节点部署',
                    'current_load': seg['avg_probability'],
                    'suggested_nodes': min(5, int(seg['avg_probability'] * 10))
                })
            elif seg['avg_probability'] < threshold_low:
                # 低压时段，建议减少节点
                recommendations.append({
                    'time_period': f"{seg['start_time']} - {seg['end_time']}",
                    'recommendation': '减少节点部署',
                    'current_load': seg['avg_probability'],
                    'suggested_nodes': max(1, int(seg['avg_probability'] * 3))
                })
            else:
                recommendations.append({
                    'time_period': f"{seg['start_time']} - {seg['end_time']}",
                    'recommendation': '保持当前节点',
                    'current_load': seg['avg_probability'],
                    'suggested_nodes': 3  # 默认节点数
                })
        
        return recommendations


# 示例使用
def demo_time_series_clustering():
    """时间序列聚类演示"""
    print("=" * 80)
    print("基于时间序列聚类的室内设计客户管理平台 - 算法演示")
    print("=" * 80)
    
    # 1. 创建模拟数据
    np.random.seed(42)
    n_points = 500
    
    # 生成时间序列（模拟一天内的访问量）
    timestamps = pd.date_range('2023-05-20 09:00', periods=n_points, freq='1min')
    
    # 模拟访问量：上午高峰、中午低谷、下午高峰
    visit_pattern = np.zeros(n_points)
    
    # 上午高峰 (9:00-12:00)
    morning_mask = (timestamps.hour >= 9) & (timestamps.hour < 12)
    visit_pattern[morning_mask] = np.random.normal(0.8, 0.1, morning_mask.sum())
    
    # 中午低谷 (12:00-14:00)
    noon_mask = (timestamps.hour >= 12) & (timestamps.hour < 14)
    visit_pattern[noon_mask] = np.random.normal(0.3, 0.1, noon_mask.sum())
    
    # 下午高峰 (14:00-18:00)
    afternoon_mask = (timestamps.hour >= 14) & (timestamps.hour < 18)
    visit_pattern[afternoon_mask] = np.random.normal(0.7, 0.15, afternoon_mask.sum())
    
    # 创建数据框
    df = pd.DataFrame({
        'timestamp': timestamps,
        'visit_count': visit_pattern * 100  # 放大到实际访问量
    })
    
    # 2. 初始化聚类平台
    platform = TimeSeriesClusteringPlatform(n_clusters_range=(2, 8))
    
    # 3. 预处理数据
    print("\n1. 数据预处理...")
    features = platform.preprocess_data(df.set_index('timestamp'))
    
    # 4. 寻找最优聚类数量
    print("\n2. 寻找最优聚类数量...")
    optimal_n = platform.find_optimal_clusters(features)
    
    # 5. 执行模糊分段
    print("\n3. 执行时间序列模糊分段...")
    segments = platform.fuzzy_segmentation(features, timestamps)
    
    # 6. 可视化结果
    print("\n4. 生成可视化结果...")
    platform.plot_clustering_results(features, timestamps, segments)
    
    # 7. 分布式平台动态扩缩容建议
    print("\n5. 分布式平台动态扩缩容建议:")
    print("-" * 80)
    
    distributed_platform = DistributedPlatform()
    distributed_platform.init_redis()
    
    recommendations = distributed_platform.dynamic_scaling(0.5, segments)
    
    for rec in recommendations:
        print(f"时间段: {rec['time_period']}")
        print(f"  负载水平: {rec['current_load']:.2%}")
        print(f"  建议操作: {rec['recommendation']}")
        print(f"  建议节点数: {rec['suggested_nodes']}")
        print("-" * 40)
    
    return platform, segments, recommendations


# 运行演示
if __name__ == "__main__":
    demo_time_series_clustering()