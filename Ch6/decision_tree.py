from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz, DecisionTreeRegressor

iris = load_iris()
X = iris.data[:, 2:] # petal length and width
y = iris.target

tree_clf = DecisionTreeClassifier(max_depth=2)
tree_clf.fit(X, y)

# visualize the graph
export_graphviz(
tree_clf,
out_file= r"C:\Users\aliak\OneDrive - HEC Montréal\Git_local\learning\Ch6\iris_tree.dot",
feature_names= iris.feature_names[2:],
class_names= iris.target_names,
rounded=True,
filled=True
)
# Then to convert the "iris_tree.dot" file in the out_file path use this command in cmd line
# first go to the address of out_file then type:
# dot -Tpng iris_tree.dot -o iris_tree.png


# Decision tree regression

tree_reg = DecisionTreeRegressor(max_depth=2)
tree_reg.fit(X, y)

