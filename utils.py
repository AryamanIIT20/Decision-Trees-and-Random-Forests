import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
pi = np.pi


colors = [ "red", "blue", "green", "yellow", "purple", "orange", "black" ]

class BinaryDecisionTree():
    class TreeNode():
        def __init__(self, X, y):
            self.X = X
            self.y = y
            self.feature = None
            self.threshold = None
            self.pmf = None
            self.entropy = None
            self.left = None
            self.right = None
            self.IG = None
            self.is_leaf = False
            self.depth = None
            
    def __init__(self,X,y, num_thresholds, min_leaf):
        num_features = X.shape[1]
        labels = set(y)
        num_classes = len(labels)
        self.X = X
        self.y = y
        self.num_thresholds = num_thresholds
        self.min_leaf = min_leaf
        self.num_features = num_features
        self.num_classes = num_classes
        self.root = None
        self.num_nodes = 1
        self.num_leaves = 0
        
    def get_pmf_entropy(self,y):
        num_classes = self.num_classes
        labels = range(num_classes)
        pmf = np.zeros(num_classes)
        for i in range(num_classes):
            pmf[i] = len(y[y==labels[i]])
        pmf = pmf/len(y)
        entropy = 0
        for p in pmf:
            if p!=0:
                entropy += -p*np.log2(p)

        return pmf,entropy
        
    def split_node(self,node):
        X = node.X
        y = node.y
        min_features = X.min(axis = 0)
        max_features = X.max(axis = 0)
        num_thresholds,num_features = self.num_thresholds,self.num_features
        thresholds = np.zeros((num_thresholds,num_features))
        for i in range(num_features):
            thresholds[:,i] = np.random.uniform(min_features[i], max_features[i],num_thresholds)
        
        IG = np.zeros((num_thresholds, num_features))
        num_samples = y.shape[0]
        pmf,entropy = node.pmf,node.entropy
        for i in range(num_features):
            for j in range(num_thresholds):
                threshold = thresholds[j,i]
                X_left = node.X[node.X[:,i]<threshold]
                y_left = node.y[node.X[:,i]<threshold]
                X_right = node.X[node.X[:,i]>=threshold]
                y_right = node.y[node.X[:,i]>=threshold]
                num_left = y_left.shape[0]
                p_left = num_left/num_samples
                p_right = 1 - p_left
                _,entropy_left = self.get_pmf_entropy(y_left)
                _,entropy_right = self.get_pmf_entropy(y_right)
                IG[j,i] = entropy - (entropy_left*p_left + entropy_right*p_right)
    
        ind = np.unravel_index(np.argmax(IG, axis=None), IG.shape)  
        optimum_threshold = thresholds[ind]
        optimum_feature = ind[1]
        optimum_IG = IG[ind]
        
        X_left = X[X[:,optimum_feature]<optimum_threshold]
        y_left = y[X[:,optimum_feature]<optimum_threshold]
        X_right = X[X[:,optimum_feature]>=optimum_threshold]
        y_right = y[X[:,optimum_feature]>=optimum_threshold]
    
        results = {"pmf":pmf,
                   "entropy":entropy,
                   "optimum_threshold":optimum_threshold,
                   "optimum_feature":optimum_feature,
                   "IG":optimum_IG,
                   "X_left":X_left,
                   "X_right":X_right,
                   "y_left":y_left,
                   "y_right":y_right
                  }
        
        return results

    def train_tree_helper(self, node, depth):
        X = node.X
        y = node.y
        node.depth = depth
        pmf,entropy = self.get_pmf_entropy(y)
        node.pmf = pmf
        node.entropy = entropy
        num_samples = X.shape[0]
        if num_samples <= self.min_leaf or node.entropy == 0:
            node.is_leaf = True
            self.num_leaves+=1
            return
        results = self.split_node(node)
        X_left = results["X_left"]
        X_right = results["X_right"]
        y_left = results["y_left"]
        y_right = results["y_right"]
        node.threshold = results["optimum_threshold"]
        node.feature = results["optimum_feature"]
        node.left = self.TreeNode(X_left,y_left)
        node.right = self.TreeNode(X_right, y_right)
        
        node.IG = results["IG"]
        self.num_nodes+=2
        self.train_tree_helper(node.left, depth+1)
        self.train_tree_helper(node.right, depth+1)
        
    def train_tree(self):
        X_train = self.X
        y_train = self.y
        self.root = self.TreeNode(X_train,y_train)
        self.train_tree_helper(self.root,0)

    def display_tree_helper(self,node):
        labels = list(np.arange(self.num_classes))
        if node is None:
            return
        if not node.is_leaf:
            IG = node.IG
            feature = node.feature
            threshold = node.threshold
            print(f"Depth: {node.depth} | x[{feature}]<{threshold:.2f}? | IG : {IG:.3f} | Entropy: {node.entropy:.3f}")
            self.display_tree_helper(node.left)
            self.display_tree_helper(node.right)
        else:
            pmf = node.pmf
            print(f"Depth: {node.depth} | Leaf Node | Entropy : {node.entropy:.3f} | Distribution : {node.pmf}")
            # plt.figure()
            # plt.bar(labels, pmf, color = colors[0:self.num_classes])
            # plt.title(f"Leaf node")
            # plt.show()

    def display_tree(self):
        node = self.root
        self.display_tree_helper(node)
        
    def infer_tree(self,X):
        preds = []
        if len(X.shape)==1:
            X = np.expand_dims(X, axis = 0)
            
        for sample in X:
            node = self.root
            pmf = None
            while True:
                if node.is_leaf:
                    preds.append(np.argmax(node.pmf))
                    pmf = node.pmf
                    break
                feature = node.feature
                threshold = node.threshold
                # print(f"Sample in tree: {sample}")
                # print(f"feature: {feature}")
                # print(f"threshold: {threshold}")
                if sample[feature]<threshold:
                    node = node.left
                else:
                    node = node.right

        preds = np.array(preds)
        return preds,pmf


def plot_decision_boundary_tree(model, X, y):
    
    # Setup prediction boundaries and grid
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    h = 200
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, h), np.linspace(y_min, y_max, h))
    
    X_to_pred_on = (np.column_stack((xx.ravel(), yy.ravel())))
    y_pred,_ = model.infer_tree(X_to_pred_on)

    y_pred = y_pred.reshape(xx.shape)
    plt.contourf(xx, yy, y_pred, cmap="jet", alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=20, cmap="jet")
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("Decision Boundary")
    plt.axis(False)



class RandomForest():
    def __init__(self,X,y,num_trees, num_thresholds, min_leaf):
        num_features = X.shape[1]
        labels = set(y)
        num_classes = len(labels)
        self.X = X
        self.y = y
        self.num_thresholds = num_thresholds
        self.min_leaf = min_leaf
        self.num_features = num_features
        self.num_classes = num_classes
        self.num_trees = num_trees
        self.X_bootstrap = None
        self.y_bootstrap = None
        self.trees = []
        
    def train_forest(self):
        self.data_bootstrap()
        for i in range(self.num_trees):
            Xi,yi = self.X_bootstrap[i], self.y_bootstrap[i]
            tree = BinaryDecisionTree(Xi,yi,self.num_thresholds, self.min_leaf)
            tree.train_tree()
            self.trees.append(tree)

    def get_pmf_entropy(self,y):
        num_classes = self.num_classes
        labels = range(num_classes)
        pmf = np.zeros(num_classes)
        for i in range(num_classes):
            pmf[i] = len(y[y==labels[i]])
        pmf = pmf/len(y)
        entropy = 0
        for p in pmf:
            if p!=0:
                entropy += -p*np.log2(p)

        return pmf,entropy
    
    def display_bootstrap(self):
        labels = list(np.arange(self.num_classes))
        pmf,entropy = self.get_pmf_entropy(self.y)
        plt.figure()
        plt.bar(labels,pmf,color = colors[0:self.num_classes])
        plt.title(f"Original Dataset Entropy: {entropy:.3f}")
        plt.show()
        for i in range(self.num_trees):
            y = self.y_bootstrap[i]
            pmf,entropy = self.get_pmf_entropy(y)
            plt.figure()
            plt.bar(labels,pmf,color = colors[0:self.num_classes])
            plt.title(f"Bag {i} Entropy: {entropy:.3f}")
            plt.show()
        
    
    def data_bootstrap(self):
        X_bootstrap = {}
        y_bootstrap = {}
        num_trees, num_classes = self.num_trees, self.num_classes
        for i in range(num_trees):
            X_bootstrap[i] = []
            y_bootstrap[i] = []
    
        for i in range(num_classes):
            Xi = self.X[self.y==i]
            Ni = Xi.shape[0]
            assignment = np.random.choice(np.arange(num_trees),size = Ni)
            for j in range(Ni):
                X_bootstrap[assignment[j]].append(Xi[j])
                y_bootstrap[assignment[j]].append(i)

        for i in range(num_trees):
            X_bootstrap[i] = np.stack(X_bootstrap[i])
            y_bootstrap[i] = np.array(y_bootstrap[i])

        self.X_bootstrap = X_bootstrap
        self.y_bootstrap = y_bootstrap
            
    def infer_forest(self,X):
        preds = []
        for sample in X:
            # print(f"Sample in forest: {sample}")
            pmf = np.zeros((self.num_classes))
            pred = 0.0
            for tree in self.trees:
                _,p = tree.infer_tree(sample)
                pmf = pmf + p
            pmf/=self.num_trees
            pred = np.argmax(pmf)
            preds.append(pred)
            
        preds = np.array(preds)
        return preds



def plot_decision_boundary_forest(model, X, y):
    
    # Setup prediction boundaries and grid
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    h = 100
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, h), np.linspace(y_min, y_max, h))
    
    X_to_pred_on = (np.column_stack((xx.ravel(), yy.ravel())))
    y_pred = model.infer_forest(X_to_pred_on)

    y_pred = y_pred.reshape(xx.shape)
    plt.contourf(xx, yy, y_pred, cmap="jet", alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=20, cmap="jet")
    plt.title("Decision Boundary")
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.axis(False)
