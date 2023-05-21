# username - complete info
# id1      - complete info
# name1    - complete info
# id2      - complete info
# name2    - complete info


"""A class represnting a node in an AVL tree"""


class AVLNode(object):
    """Constructor, you are allowed to add more fields.

    @type key: int or None
    @param key: key of your node
    @type value: any
    @param value: data of your node
    """

    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.left = None
        self.right = None
        self.parent = None
        self.height = -1
        self.size = 0

    def create_leaf_with_virtual_nodes(key, value):
        leaf = AVLNode(key, value)
        leaf.height(0)
        leaf.size(1)
        leaf.left = AVLNode(None, None)
        leaf.left.parent = leaf
        leaf.right = AVLNode(None, None)
        leaf.right.parent = leaf

        return leaf

    def compute_size_and_update_property(self):
        right_child_size = self.right.size
        left_child_size = self.left.size

        self.size = right_child_size + left_child_size + 1

    """
    Computes the balance factor of the node.

    This method calculates the balance factor of the node also updates the height and size properties of the node.
    
    Returns:
    - balance_factor (int): The balance factor of the node.
    """

    def compute_balance_factor(self):
        if self == None or (not self.is_real_node()):
            return 0

        right_child_height = self.right.height
        left_child_height = self.left.height
        balance_factor = left_child_height - right_child_height

        self.height = max(right_child_height, left_child_height) + 1

        self.compute_size_and_update_property()

        return balance_factor

    """
    Checks if a node is empty.

    Parameters:
        node: A node in the tree.

    Returns:
        bool: True if the node is empty, False otherwise.
    """
    def is_empty_node(node):
        return node == None or (not node.is_real_node())

    """
    Checks if the node is a leaf node (a node with 2 'virtual node' childs is considered a leaf as well).

    Returns:
        bool: True if the node is a leaf node, False otherwise.
    """

    def is_leaf(self):
        right_child = self.right
        left_child = self.left

        is_right_child_empty = self.is_empty_node(right_child)
        is_left_child_empty = self.is_empty_node(left_child)

        return is_right_child_empty and is_left_child_empty

    """
    Checks whether the node has exactly one child.
    
    Returns:
        bool: True if the node has exactly one child, False otherwise.
    """

    def has_one_child(self):
        right_child = self.right
        left_child = self.left

        is_right_child_empty = self.is_empty_node(right_child)
        is_left_child_empty = self.is_empty_node(left_child)

        return (is_right_child_empty and (not is_left_child_empty)) or \
            (is_left_child_empty and (not is_right_child_empty))

    """
    Checks whether the given node is the left child of its parent.
    Returns:
        bool: True if the node is the left child of its parent, False otherwise.
    """

    def is_left_child_of_parent(self):
        parent = self.parent
        return parent.left == self

    """
    Updates the parent's child node reference with a new child node.

    Parameters:
        old_child: The old child node to be replaced.
        new_child: The new child node to replace the old child.
    """

    def update_parents_child(self, old_child, new_child):
        if old_child.is_left_child_of_parent():
            self.set_left(new_child)
        else:
            self.set_right(new_child)

    """returns the key

	@rtype: int or None
	@returns: the key of self, None if the node is virtual
	"""

    def get_key(self):
        return self.key

    """returns the value

	@rtype: any
	@returns: the value of self, None if the node is virtual
	"""

    def get_value(self):
        return self.value

    """returns the left child
	@rtype: AVLNode
	@returns: the left child of self, None if there is no left child (if self is virtual)
	"""

    def get_left(self):
        return self.left

    """returns the right child

	@rtype: AVLNode
	@returns: the right child of self, None if there is no right child (if self is virtual)
	"""

    def get_right(self):
        return self.right

    """returns the parent

	@rtype: AVLNode
	@returns: the parent of self, None if there is no parent
	"""

    def get_parent(self):
        return self.parent

    """returns the height

	@rtype: int
	@returns: the height of self, -1 if the node is virtual
	"""

    def get_height(self):
        return self.height

    """returns the size of the subtree

	@rtype: int
	@returns: the size of the subtree of self, 0 if the node is virtual
	"""

    def get_size(self):
        return self.size

    """sets key

	@type key: int or None
	@param key: key
	"""

    def set_key(self, key):
        self.key = key

    """sets value

	@type value: any
	@param value: data
	"""

    def set_value(self, value):
        self.value = value

    """sets left child and updates it's parent to be the new one

	@type node: AVLNode
	@param node: a node
	"""

    def set_left(self, node):
        self.left = node
        node.set_parent(self)

    """sets right child and updates it's parent to be the new one

	@type node: AVLNode
	@param node: a node
	"""

    def set_right(self, node):
        self.right = node
        node.set_parent(self)

    """sets parent.
    Additionally, the method updates the height and size properties
    of the new parent node based on the heights and sizes of its left and right child nodes.

	@type node: AVLNode
	@param node: a node
	"""

    def set_parent(self, node):
        if node == None:
            self.parent = None
        else:
            new_parent: AVLNode = node
            self.parent = new_parent

            parent_left = new_parent.left
            parent_right = new_parent.right

            new_height = max(parent_left.height, parent_right.height) + 1
            new_parent.set_height(new_height)

            new_size = parent_left.size + parent_right.size + 1
            new_parent.set_size(new_size)

    """sets the height of the node

	@type h: int
	@param h: the height
	"""

    def set_height(self, h):
        self.height = h

    """sets the size of node

	@type s: int
	@param s: the size
	"""

    def set_size(self, s):
        self.size = s

    """returns whether self is not a virtual node

	@rtype: bool
	@returns: False if self is a virtual node, True otherwise.
	"""

    def is_real_node(self):
        return not self.key == None


"""
A class implementing an AVL tree.
"""


class AVLTree(object):

    """
    Constructor, you are allowed to add more fields.

    """

    def __init__(self):
        self.root = None

    """Sets the root node of the AVL tree.

    Parameters:
        node: The node to be set as the root.
    """

    def set_root(self, node):
        self.root = node
        node.set_parent(None)

    """Performs a right rotation on the given `old_root` node in the AVL tree.
    The right rotation promotes 'new_root' (the left child of `old_root`) in 'old_root''s place,
    while `old_root` becomes the right child of the new root.
    
    Args:
        old_root (AVLNode): The node to perform the right rotation on.

    Complexity:
        The rotation operation takes constant time O(1).
    """

    def rotate_right(self, old_root: AVLNode):
        new_root: AVLNode = old_root.get_left()
        new_root_right = new_root.get_right()

        old_root_parent = old_root.get_parent()
        if old_root_parent != None:
            old_root_parent.update_parents_child(old_root, new_root)
        else:
            self.set_root(new_root)

        old_root.set_left(new_root_right)
        new_root.set_right(old_root)

    """Performs a left rotation on the given `old_root` node in the AVL tree.
    The left rotation promotes 'new_root' (the right child of `old_root`) in 'old_root''s place,
    while `old_root` becomes the left child of the new root.
    
    Args:
        old_root (AVLNode): The node to perform the left rotation on.

    Complexity:
        The rotation operation takes constant time O(1).
    """

    def rotate_left(self, old_root: AVLNode):
        new_root: AVLNode = old_root.get_right()
        new_root_left = new_root.get_left()

        old_root_parent = old_root.get_parent()
        if old_root_parent != None:
            old_root_parent.update_parents_child(old_root, new_root)
        else:
            self.set_root(new_root)

        old_root.set_right(new_root_left)
        new_root.set_left(old_root)

    """Performs a left_rotation on 'old_root's left child and then a right rotation on 'old_root'.
    
    Args:
        old_root (AVLNode): The root of the subtree to perform the rotations on.

    Complexity:
        The rotation operation takes constant time O(1).
    """

    def rotate_left_then_right(self, old_root: AVLNode):
        old_root_left_child: AVLNode = old_root.get_left()

        self.rotate_left(old_root_left_child)
        self.rotate_right(old_root)

    """Performs a right rotation on 'old_root's right child and then a left rotation on 'old_root'.
    
    Args:
        old_root (AVLNode): The root of the subtree to perform the rotations on.

    Complexity:
        The rotation operation takes constant time O(1).
    """

    def rotate_right_then_left(self, old_root: AVLNode):
        old_root_right_child: AVLNode = old_root.get_right()
        self.rotate_right(old_root_right_child)
        self.rotate_left(old_root)

    """Finds the successor node for a given `node` in the AVL tree when the node has two children.

    The algorithm starts by moving to the right child of the given `node`, and then iteratively
    moves to the left child of each subsequent node until it reaches the leftmost descendant of
    the right child (which will be the successor node). It returns the parent of this successor node.

    Args:
        node (AVLNode): The node for which to find the successor.

    Returns:
        AVLNode: The parent node of the successor node.

    Complexity:
        The operation has a time complexity of O(log n) in the worst case, where n is the number of nodes
        in the AVL tree.
    """

    def find_successor_for_node_with_two_childs(self, node: AVLNode):
        successor_contestant: AVLNode = node.get_right()
        while not successor_contestant.is_empty_node(successor_contestant):
            successor_contestant = successor_contestant.get_left()
        return successor_contestant.get_parent()

    """searches for a value in the dictionary corresponding to the key

	@type key: int
	@param key: a key to be searched
	@rtype: any
	@returns: the value corresponding to key.
    
    Complexity: 
        The operation has a time complexity of O(log n) in the worst case, where n is the number of nodes
        in the AVL tree. 
	"""

    def search(self, key):
        root = self.root

        while root != None and root.is_real_node():
            root_key = root.get_key()
            if root_key == key:
                return root
            if root_key < key:
                root = root.get_right()
            if root_key > key:
                root = root.get_left()

        return None

    """inserts val at position i in the dictionary

	@type key: int
	@pre: key currently does not appear in the dictionary
	@param key: key of item that is to be inserted to self
	@type val: any
	@param val: the value of the item
	@rtype: int
	@returns: the number of rebalancing operation due to AVL rebalancing

    Complexity: 
        The operation has a time complexity of O(log n) in the worst case, where n is the number of nodes
        in the AVL tree. 
	"""

    def insert(self, key, val):
        leaf_for_insert = AVLNode.create_leaf_with_virtual_nodes(key, val)

        if self.root == None:
            self.set_root(leaf_for_insert)
            return 0

        parent_of_leaf = self.physical_insert(leaf_for_insert)

        return self.fix_tree(parent_of_leaf)

    """Finds the parent node where a new node with the given key should be inserted.

    Args:
        key: The key of the new node to be inserted.

    Returns:
        AVLNode: The parent node where the new node should be inserted.

    Complexity:
        The time complexity of finding the parent for insertion is O(logn) in the worst case,
        where 'n' is the number of nodes in the tree.

    """

    def find_parent_for_insert(self, key):
        root: AVLNode = self.root

        while root.is_real_node():
            root_key = root.get_key()

            if root_key < key:
                root = root.get_right()
            if root_key > key:
                root = root.get_left()

        return root.get_parent()

    """Finds the parent node that has an illegal balance factor in its subtree.

    Starting from the given node, the function traverses up the tree towards the root,
    searching for the first node that has an absolute balance factor greater than 1.
    It stops the traversal if it reaches the root or encounters a node with a legal balance factor.

    Args:
        node (AVLNode): The starting node for the search.

    Returns:
        AVLNode: The parent node with an illegal balance factor, or None if no such node is found.

    Complexity:
        The time complexity of finding the parent with an illegal balance factor is O(logn) in the worst case,
        where 'n' is the number of nodes in the tree.

    """

    def find_parent_with_illegal_balance_factor(self, node: AVLNode):
        while node != None and abs(node.compute_balance_factor()) <= 1:
            node = node.get_parent()
        return node

    """Fixes the subtree rooted at the given illegal root node.

    The function performs necessary rotationsto restore the balance of the subtree rooted at the illegal root node.
    It determines the rotations needed based on the balance factor of the illegal root and its child nodes.

    Args:
        illegal_root (AVLNode): The illegal root node with an imbalance in its subtree.

    Returns:
        int: The number of rotations made during the fix.

    Complexity:
        The time complexity of fixing the subtree is O(1).

    """

    def fix_tree_of_illegal_root(self, illegal_root):
        if illegal_root == None:
            return 0

        illegal_balance_factor = illegal_root.compute_balance_factor()

        # Rotate according to the balance factor, as seen in the lecture
        if illegal_balance_factor == -2:
            right_child_balance_factor = illegal_root.get_right().compute_balance_factor()
            if right_child_balance_factor in [-1, 0]:
                self.rotate_left(illegal_root)
                return 1
            if right_child_balance_factor == 1:
                self.rotate_right_then_left(illegal_root)
                return 2
        else:  # illegal_balance_factor == 2
            left_child_balance_factor = illegal_root.get_left().compute_balance_factor()
            if left_child_balance_factor in [1, 0]:
                self.rotate_right(illegal_root)
                return 1
            if left_child_balance_factor == -1:
                self.rotate_left_then_right(illegal_root)
                return 2

    """Fixes the AVL tree up to the root starting from the given node.

    The function iteratively fixes the tree from the given node up to the root by identifying and resolving nodes with illegal balance factors.

    Args:
        node (AVLNode): The node to fix the tree from.

    Returns:
        int: The sum of the rotations made to fix the tree.

    Complexity:
        The time complexity of fixing the tree is O(log n), where n is the number of nodes in the AVL tree.
    """

    def fix_tree(self, node):
        sum = 0

        # Fix the tree from the given node up to the root
        while node != None:
            node = self.find_parent_with_illegal_balance_factor(node)
            sum += self.fix_tree_of_illegal_root(node)

        return sum

    """
    Inserts a new leaf node into the AVL tree.

    The function determines the appropriate parent node for insertion based on the key value of the leaf node.

    Args:
        leaf_for_insert (AVLNode): The leaf node to be inserted into the AVL tree.

    Returns:
        AVLNode: The parent node where the leaf node is inserted.

    Note:
        This function assumes that the leaf node is not already present in the tree.

    Complexity:
        The time complexity of the physical insertion operation is O(log n), where n is the number of nodes in the AVL tree.
    """

    def physical_insert(self, leaf_for_insert: AVLNode):
        key = leaf_for_insert.get_key()
        parent_for_insert = self.find_parent_for_insert(key)
        parent_for_insert_key = parent_for_insert.get_key()

        if parent_for_insert_key > key:
            parent_for_insert.set_left(leaf_for_insert)
        if parent_for_insert_key < key:
            parent_for_insert.set_right(leaf_for_insert)

        return parent_for_insert

    """deletes node from the dictionary

	@type node: AVLNode
	@pre: node is a real pointer to a node in self
	@rtype: int
	@returns: the number of rebalancing operation due to AVL rebalancing

    Complexity:
        The operation has a time complexity of O(log n) in the worst case, where n is the number of nodes
        in the AVL tree. 
	"""

    def delete(self, node: AVLNode):
        node_to_fix_from = None

        if node.is_leaf() or node.has_one_child():
            node_to_fix_from = self.physical_delete(node)
        else:
            successor = self.find_successor_for_node_with_two_childs(node)
            node_to_fix_from = self.physical_delete(successor)
            self.replace_node_in_tree(node, successor)

        return self.fix_tree(node_to_fix_from)

    """Physically deletes a node from the AVL tree by adjusting the tree structure.

    The function handles the physical deletion of a node based on its type:
    - If the node is a leaf or has one child, its child is assigned as the new child of its parent.
    - If the node is the root, one of it's children become the new root with preference to the left child
      (if the node is a leaf as well then the new root will point to None).

    Args:
        node (AVLNode): The node to be physically deleted.

    Returns:
        AVLNode: The parent node from which tree balancing should start.

    Complexity:
        The time complexity of physical deletion is O(1).

    """

    def physical_delete(self, node: AVLNode):
        child = None
        left_child = node.get_left()
        if left_child.is_real_node():
            child = left_child
        else:
            child = node.get_right()

        node_parent = node.get_parent()
        if node_parent == None:
            if child.is_real_node():
                self.set_root(child)
            else:
                self.root = None
        else:
            if node.is_left_child_of_parent():
                node_parent.set_left(child)
            else:
                node_parent.set_right(child)
            node.parent = None

        return node_parent

    """Replaces a node in the AVL tree with a new node.

    The function replaces the old node with the new node while preserving the old node's left and right children.
    If the old node is the root, the new node becomes the root.
   
    Args:
        old_node (AVLNode): The node to be replaced.
        new_node (AVLNode): The new node to replace the old node.

    Complexity:
        The time complexity of replacing a node in the tree is O(1).

    """

    def replace_node_in_tree(self, old_node: AVLNode, new_node: AVLNode):
        if self.root == old_node:
            self.set_root(new_node)

        old_node_parent = old_node.get_parent()
        new_node.set_left(old_node.get_left())
        new_node.set_right(old_node.get_right())
        old_node_parent.update_parents_child(old_node, new_node)

    """returns an array representing dictionary 

	@rtype: list
	@returns: a sorted list according to key of touples (key, value) representing the data structure

    Complexity:
        The operation has a time complexity of O(n) in the worst case, where n is the number of nodes
        in the AVL tree. 

	"""

    def avl_to_array(self):
        array = []
        self.avl_to_array_rec(self.root, array)
        return array

    """
    Recursively converts an AVL tree into an array representation.

    The function traverses the AVL tree in an in-order fashion (left-subtree, root, right-subtree)
    and appends each node's key-value pair to the provided array.

    Args:
        node (AVLNode): The current node being processed.
        array (list): The array to store the key-value pairs of the AVL tree.

    Complexity:
        The time complexity of this function is O(n), where 'n' is the number of nodes in the AVL tree.
    """

    def avl_to_array_rec(self, node, array):
        if AVLNode.is_empty_node(node):
            return
        self.avl_to_array_rec(node.get_left(), array)
        array.append((node.get_key(), node.get_value()))
        self.avl_to_array_rec(node.get_right(), array)

    """returns the number of items in dictionary 

	@rtype: int
	@returns: the number of items in dictionary 
	"""

    def size(self):
        return self.get_root().get_size()

    """splits the dictionary at a given node

	@type node: AVLNode
	@pre: node is in self
	@param node: The intended node in the dictionary according to whom we split
	@rtype: list
	@returns: a list [left, right], where left is an AVLTree representing the keys in the 
	dictionary smaller than node.key, right is an AVLTree representing the keys in the 
	dictionary larger than node.key.
	"""

    # O(logn)
    def split(self, node: AVLNode):
        greater_array_of_node_tuples = []
        lower_array_of_node_tuples = []
        greater_than_node_basis = node.get_right()
        greater_than_tree_basis = self.create_tree(greater_than_node_basis)
        lower_than_node_basis = node.get_left()
        lower_than_tree_basis = self.create_tree(lower_than_node_basis)
        parent = node.get_parent()

        # Create tuples with connecting node and trees to connect, as seen in the lecture
        while parent != None:
            if not node.is_left_child_of_parent():
                lower_array_of_node_tuples.append((parent, parent.get_left()))
            else:
                greater_array_of_node_tuples.append(
                    (parent, parent.get_right()))
            node = parent
            parent = parent.get_parent()

        lower_than_tree_basis.join_tree_with_array_of_node_tuples(
            lower_array_of_node_tuples)
        greater_than_tree_basis.join_tree_with_array_of_node_tuples(
            greater_array_of_node_tuples)

        return [lower_than_tree_basis, greater_than_tree_basis]

    # O(logn)
    def join_tree_with_array_of_node_tuples(self, array_of_node_tuples):
        for mid_node, other_node in array_of_node_tuples:
            other_tree = self.create_tree(other_node)
            self.join(other_tree, mid_node.get_key(), mid_node.get_value())

    """joins self with key and another AVLTree

	@type tree: AVLTree 
	@param tree: a dictionary to be joined with self
	@type key: int 
	@param key: The key separting self with tree
	@type val: any 
	@param val: The value attached to key
	@pre: all keys in self are smaller than key and all keys in tree are larger than key,
	or the other way around.
	@rtype: int
	@returns: the absolute value of the difference between the height of the AVL trees joined
	"""

    # O(logn)
    def join(self, tree, key, val):
        # Find the tree with the lower values (t1) and the tree with the higher values (t2)
        if not self.root.is_real_node() and not tree.root.is_real_node():
            self.set_root(AVLNode.create_leaf_with_virtual_nodes(key, val))
            return 1
        if self.root.is_real_node():
            if self.root.get_key() < key:
                t1 = self
                t2 = tree
            else:
                t1 = tree
                t2 = self
        else:
            if tree.root.get_key() > key:
                t1 = self
                t2 = tree
            else:
                t1 = tree
                t2 = self

        t1_height = t1.root.get_height()
        t2_height = t2.root.get_height()
        height_delta = abs(t1_height - t2_height) + 1
        x = AVLNode.create_leaf_with_virtual_nodes(key, val)

        if t1_height == t2_height:
            self.join_trees_with_equal_heights(t1, t2, x)
            return height_delta

        if t1_height < t2_height:
            self.join_trees_left_tree_is_smaller(t1, t2, x)
            return height_delta

        self.join_trees_right_tree_is_smaller(t1, t2, x)

        return height_delta

    # O(1)
    def create_tree(self, root):
        tree = AVLTree()
        tree.set_root(root)
        return tree

    # O(1)
    def join_trees_with_equal_heights(self, t1, t2, x):
        x.set_left(t1.root)
        x.set_right(t2.root)
        self.root = x

    # O(logn)
    def join_trees_left_tree_is_smaller(self, t1, t2, x):
        t2_node = t2.root

        # Find a node with height h or h-1 on left-most node trail of t2
        while t2_node.get_height() > t1.root.get_height():
            t2_node = t2_node.get_left()

        b = t2_node
        b_parent = b.get_parent()

        # "Physical" join of the trees as seen in the lecture
        x.set_left(t1.root)
        x.set_right(b)
        b_parent.set_left(x)

        t2.fix_tree(x)

        self.root = t2.root

    # O(logn)
    def join_trees_right_tree_is_smaller(self, t1, t2, x):
        t1_node = t1.root

        # Find a node with height h or h-1 on right-most node trail of t1
        while t1_node.get_height() > t2.root.get_height():
            t1_node = t1_node.get_right()

        b = t1_node
        b_parent = b.get_parent()

        # "Physical" join of the trees as seen in the lecture
        x.set_left(b)
        x.set_right(t2.root)
        b_parent.set_right(x)

        t1.fix_tree(x)

        self.root = t1.root

    """compute the rank of node in the self

	@type node: AVLNode
	@pre: node is in self
	@param node: a node in the dictionary which we want to compute its rank
	@rtype: int
	@returns: the rank of node in self
	"""

    # O(logn)
    def rank(self, node):
        current_node: AVLNode = self.root
        current_rank = 1

        # Find the node using regular search, calculate the rank on the way
        while current_node.get_key() != node.get_key():
            left_node = current_node.get_left()

            if current_node.get_key() < node.get_key():
                current_rank += (left_node.get_size() + 1)
                current_node = current_node.get_right()
            else:  # current_node.get_key() > node.get_key()
                current_node = current_node.get_left()

        return current_rank + node.get_left().get_size()

    """finds the i'th smallest item (according to keys) in self

	@type i: int
	@pre: 1 <= i <= self.size()
	@param i: the rank to be selected in self
	@rtype: int
	@returns: the item of rank i in self
	"""

    # O(logn)
    def select(self, i):
        current_node: AVLNode = self.root
        current_rank = 1

        # Calculate the rank of the current node,
        # then continue left if it is too large or right if it is too small
        while current_rank <= i:
            left_node = current_node.get_left()
            total_rank = current_rank + left_node.get_size()

            if total_rank == i:
                return current_node
            if total_rank > i:
                current_node = left_node
            else:  # current_rank + left_node.get_size() < i
                current_rank = total_rank + 1
                current_node = current_node.get_right()

    """returns the root of the tree representing the dictionary

	@rtype: AVLNode
	@returns: the root, None if the dictionary is empty
	"""

    # O(1)
    def get_root(self):
        return self.root
