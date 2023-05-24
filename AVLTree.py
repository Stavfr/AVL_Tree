# username - amitrockach
# id1      - 322853813
# name1    - Amit Rockach
# id2      - 208065938
# name2    - Stav Fridman


class AVLNode(object):
    """
    A class represnting a node in an AVL tree
    """

    def __init__(self, key, value):
        """Constructor, you are allowed to add more fields.

        @type key: int or None
        @param key: key of your node
        @type value: any
        @param value: data of your node
        """
        self.key = key
        self.value = value
        self.left = None
        self.right = None
        self.parent = None
        self.height = -1
        self.size = 0

    def create_leaf_with_virtual_nodes(key, value):
        """Creates AVLNode with 2 virutal children

        @type key: int or None
        @param key: key of your node
        @type value: any
        @param value: data of your node
        """
        leaf = AVLNode(key, value)
        leaf.height = 0
        leaf.size = 1
        leaf.left = AVLNode(None, None)
        leaf.left.parent = leaf
        leaf.right = AVLNode(None, None)
        leaf.right.parent = leaf

        return leaf

    def compute_size_and_update_property(self):
        """
        Computes the size of self according to his children sizes
        """
        right_child_size = self.right.size
        left_child_size = self.left.size

        self.size = right_child_size + left_child_size + 1

    def compute_balance_factor(self):
        """Computes the balance factor and also updates the height and size properties of the node.  

        @rtype: int
        @returns: self's balance factor
        """
        has_height_updated = False

        if self == None or (not self.is_real_node()):
            return 0, has_height_updated

        right_child_height = self.right.height
        left_child_height = self.left.height
        balance_factor = left_child_height - right_child_height

        new_height = max(right_child_height, left_child_height) + 1

        if self.height != new_height:
            self.height = new_height
            has_height_updated = True

        self.compute_size_and_update_property()

        return balance_factor, has_height_updated

    def is_empty_node(node):
        """Checks if node is None or 'virtual node'

        @type node: AVLNode or None
        @param node: The node to check.
        @rtype: bool
        @returns: True if node is None or 'virtual node', False otherwise
        """
        return node == None or (not node.is_real_node())

    def is_leaf(self):
        """Checks if our node is a leaf node (a node with 2 'virtual node' children is considered a leaf as well).

        @rtype: bool
        @returns: True if the node is a leaf node, False otherwise.
        """
        right_child = self.right
        left_child = self.left

        is_right_child_empty = AVLNode.is_empty_node(right_child)
        is_left_child_empty = AVLNode.is_empty_node(left_child)

        return is_right_child_empty and is_left_child_empty

    def has_one_child(self):
        """Checks whether self has exactly one child.

        @rtype: bool
        @returns: True if the node has exactly one child, False otherwise.
        """
        right_child = self.right
        left_child = self.left

        is_right_child_empty = AVLNode.is_empty_node(right_child)
        is_left_child_empty = AVLNode.is_empty_node(left_child)

        return (is_right_child_empty and (not is_left_child_empty)) or \
            (is_left_child_empty and (not is_right_child_empty))

    def is_left_child_of_parent(self):
        """Checks whether self is the left child of its parent.

        @rtype: bool
        @returns: True if the node is the left child of its parent, False otherwise.
        """
        parent = self.parent
        return parent.left == self

    def update_parents_child(self, old_child, new_child):
        """Updates the parent's child node reference with a new child node.

        @type old_child: AVLNode
        @param old_child: The old child node to be replaced.
        @type new_child: AVLNode
        @param new_child: The new child node to replace the old child.
        """
        if old_child.is_left_child_of_parent():
            self.set_left(new_child)
        else:
            self.set_right(new_child)

    def get_key(self):
        """Returns the key

        @rtype: int or None
        @returns: the key of self, None if the node is virtual
        """
        return self.key

    def get_value(self):
        """Returns the value

        @rtype: any
        @returns: the value of self, None if the node is virtual
        """
        return self.value

    def get_left(self):
        """Returns the left child

        @rtype: AVLNode
        @returns: the left child of self, None if there is no left child (if self is virtual)
        """
        return self.left

    def get_right(self):
        """Returns the right child

        @rtype: AVLNode
        @returns: the right child of self, None if there is no right child (if self is virtual)
        """
        return self.right

    def get_parent(self):
        """Returns the parent

        @rtype: AVLNode
        @returns: the parent of self, None if there is no parent
        """
        return self.parent

    def get_height(self):
        """Returns the height

        @rtype: int
        @returns: the height of self, -1 if the node is virtual
        """
        return self.height

    def get_size(self):
        """Returns the size of the subtree

        @rtype: int
        @returns: the size of the subtree of self, 0 if the node is virtual
        """
        return self.size

    def set_key(self, key):
        """Sets key

        @type key: int or None
        @param key: key
        """
        self.key = key

    def set_value(self, value):
        """Sets value

        @type value: any
        @param value: data
        """
        self.value = value

    def set_left(self, node):
        """Sets left child and updates it's parent to be the new one

        @type node: AVLNode
        @param node: a node
        """
        self.left = node
        node.set_parent(self)

    def set_right(self, node):
        """Sets right child and updates it's parent to be the new one

        @type node: AVLNode
        @param node: a node
        """
        self.right = node
        node.set_parent(self)

    def set_parent(self, node):
        """Sets parent. Additionally, the method updates the height and size properties
        of the new parent node based on the heights and sizes of its left and right child nodes.

        @type node: AVLNode
        @param node: a node
        """
        if AVLNode.is_empty_node(node):
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

    def set_height(self, h):
        """Sets the height of the node

        @type h: int
        @param h: the height
        """
        self.height = h

    def set_size(self, s):
        """Sets the size of node

        @type s: int
        @param s: the size
        """
        self.size = s

    def is_real_node(self):
        """returns whether self is not a virtual node
        @rtype: bool
        @returns: False if self is a virtual node, True otherwise.
        """
        return not self.key == None


class AVLTree(object):
    """
    A class implementing an AVL tree.
    """

    def __init__(self):
        """
        Constructor, you are allowed to add more fields.
        """
        self.root = None

    def set_root(self, node):
        """Sets the root node of the AVL tree.

        @type node: AVLNode or None
        @param node: the node to be set as the root
        """
        self.root = node
        node.set_parent(None)

    def rotate_right(self, old_root: AVLNode):
        """Performs a right rotation on the given `old_root` node in the AVL tree.

        The right rotation promotes 'new_root' (the left child of `old_root`) in 'old_root''s place,
        while `old_root` becomes the right child of the new root.

        @type old_root: AVLNode
        @param old_root: the node to perform the right rotation on.
        """
        new_root: AVLNode = old_root.get_left()
        new_root_right = new_root.get_right()

        old_root_parent = old_root.get_parent()
        if old_root_parent != None:
            old_root_parent.update_parents_child(old_root, new_root)
        else:
            self.set_root(new_root)

        old_root.set_left(new_root_right)
        new_root.set_right(old_root)

    def rotate_left(self, old_root: AVLNode):
        """Performs a left rotation on the given `old_root` node in the AVL tree.

        The left rotation promotes 'new_root' (the right child of `old_root`) in 'old_root''s place,
        while `old_root` becomes the left child of the new root.

        @type old_root: AVLNode
        @param old_root: the node to perform the left rotation on.
        """
        new_root: AVLNode = old_root.get_right()
        new_root_left = new_root.get_left()

        old_root_parent = old_root.get_parent()
        if old_root_parent != None:
            old_root_parent.update_parents_child(old_root, new_root)
        else:
            self.set_root(new_root)

        old_root.set_right(new_root_left)
        new_root.set_left(old_root)

    def rotate_left_then_right(self, old_root: AVLNode):
        """Performs a left_rotation on 'old_root's left child and then a right rotation on 'old_root'.

        @type old_root: AVLNode
        @param old_root: The root of the subtree to perform the rotations on.
        """
        old_root_left_child: AVLNode = old_root.get_left()

        self.rotate_left(old_root_left_child)
        self.rotate_right(old_root)

    def rotate_right_then_left(self, old_root: AVLNode):
        """Performs a right rotation on 'old_root's right child and then a left rotation on 'old_root'.

        @type old_root: AVLNode
        @param old_root: The root of the subtree to perform the rotations on.
        """
        old_root_right_child: AVLNode = old_root.get_right()
        self.rotate_right(old_root_right_child)
        self.rotate_left(old_root)

    def find_successor_for_node_with_two_childs(self, node: AVLNode):
        """Finds the successor node for a given `node` in the AVL tree when the node has two children.

        The function starts by moving to the right child of the given `node`, and then iteratively
        moves to the left child of each subsequent node until it reaches the leftmost descendant of
        the right child (which will be the successor node). It returns the parent of this successor node.

        @type node: AVLNode
        @pre: node has 2 children (non virtual children).
        @param node: Node to find a successor of.
        @rtype: AVLNode
        @returns: The parent of the successor node.
        @complexity: The operation has a time complexity of O(log n) in the worst case, where n is the number of nodes
        in the AVL tree.
        """
        successor_contestant: AVLNode = node.get_right()
        while not AVLNode.is_empty_node(successor_contestant):
            successor_contestant = successor_contestant.get_left()
        return successor_contestant.get_parent()

    def search(self, key):
        """Searches for a value in the dictionary corresponding to the key

        @type key: int
        @param key: a key to be searched
        @rtype: any
        @returns: the value corresponding to key.
        @complexity: The operation has a time complexity of O(log n) in the worst case, where n is the number of nodes
        in the AVL tree. 
        """
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

    def insert(self, key, val):
        """Inserts val at position i in the dictionary

        @type key: int
        @pre: key currently does not appear in the dictionary
        @param key: key of item that is to be inserted to self
        @type val: any
        @param val: the value of the item
        @rtype: int
        @returns: the number of rebalancing operation due to AVL rebalancing
        @complexity: The operation has a time complexity of O(log n) in the worst case, where n is the number of nodes
        in the AVL tree.
        """
        leaf_for_insert = AVLNode.create_leaf_with_virtual_nodes(key, val)

        if AVLNode.is_empty_node(self.root):
            self.set_root(leaf_for_insert)
            return 0

        parent_of_leaf = self.physical_insert(leaf_for_insert)

        return self.fix_tree(parent_of_leaf)

    def find_parent_for_insert(self, key):
        """Finds the parent node where a new node with the given key should be inserted.

        @type key: int
        @param key: The key of the new node to be inserted to find the parent of.
        @rtype: AVLNode
        @returns: The parent node where a new node with the given key should be inserted
        @complexity: The time complexity of finding the parent for insertion is O(logn) in the worst case,
        where 'n' is the number of nodes in the tree.
        """
        root: AVLNode = self.root

        while root.is_real_node():
            root_key = root.get_key()

            if root_key < key:
                root = root.get_right()
            if root_key > key:
                root = root.get_left()

        return root.get_parent()

    def find_parent_with_illegal_balance_factor(self, node: AVLNode):
        """Finds the parent node that has an illegal balance factor in its subtree.

        Starting from the given node, the function traverses up the tree towards the root,
        searching for the first node that has an absolute balance factor greater than 1.
        It stops the traversal if it reaches the root or encounters a node with a legal balance factor.

        @type node: AVLNode
        @param node: The starting node for the search.
        @rtype: AVLNode or None
        @returns: The parent node with an illegal balance factor, or None if no such node is found.
        @complexity: The time complexity of finding the parent with an illegal balance factor is O(logn) in the worst case,
        where 'n' is the number of nodes in the tree.
        """
        imblanace_fix_counter = 0

        while node != None:
            balance_factor, has_height_updated = node.compute_balance_factor()
            if abs(balance_factor) >= 2:
                break
            if has_height_updated:
                imblanace_fix_counter += 1
            node = node.get_parent()

        return node, imblanace_fix_counter

    def fix_tree_of_illegal_root(self, illegal_root):
        """Fixes the subtree rooted at the given illegal root node.

        The function performs necessary rotations to restore the balance of the subtree rooted at the illegal root node.
        It determines the rotations needed based on the balance factor of the illegal root and its child nodes.

        @type illegal_root: AVLNode or None
        @param illegal_root: The illegal root node with an imbalance in its subtree.
        @rtype: int
        @returns: The number of rotations made during the fix.
        """
        if illegal_root == None:
            return 0

        illegal_balance_factor, _ = illegal_root.compute_balance_factor()

        # Rotate according to the balance factor, as seen in the lecture
        if illegal_balance_factor == -2:
            right_child_balance_factor, _ = illegal_root.get_right().compute_balance_factor()
            if right_child_balance_factor in [-1, 0]:
                self.rotate_left(illegal_root)
                return 1
            if right_child_balance_factor == 1:
                self.rotate_right_then_left(illegal_root)
                return 2
        else:  # illegal_balance_factor == 2
            left_child_balance_factor, _ = illegal_root.get_left().compute_balance_factor()
            if left_child_balance_factor in [1, 0]:
                self.rotate_right(illegal_root)
                return 1
            if left_child_balance_factor == -1:
                self.rotate_left_then_right(illegal_root)
                return 2

    def fix_tree(self, node):
        """Fixes the AVL tree up to the root starting from the given node.

        The function iteratively fixes the tree from the given node up to the root by identifying and resolving nodes with illegal balance factors.
        Additionally, the function counts the number of rotations and height fixes needed for fixing the tree.

        @type node: AVLNode
        @param node: The node to fix the tree from.
        @rtype: int
        @returns: The sum of the rotations made to fix the tree.
        @complexity: The time complexity of fixing the tree is O(log n), where n is the number of nodes in the AVL tree.
        """
        sum = 0

        # Fix the tree from the given node up to the root
        while node != None:
            node, num_of_imbalance_fixes = self.find_parent_with_illegal_balance_factor(
                node)
            sum += num_of_imbalance_fixes
            sum += self.fix_tree_of_illegal_root(node)

        return sum

    def physical_insert(self, leaf_for_insert: AVLNode):
        """Inserts a new leaf node with virutal children into the AVL tree.

        The function determines the appropriate parent node for insertion based on the key value of the leaf node.

        @type leaf_for_insert: AVLNode
        @pre: The key of leaf node does not appear in the tree.
        @param leaf_for_insert:  The leaf node to be inserted into the AVL tree.
        @rtype: AVLNode
        @returns: The parent node where the leaf node is inserted.
        @complexity: The time complexity of the physical insertion operation is O(log n), where n is the number of nodes in the AVL tree.
        """
        key = leaf_for_insert.get_key()
        parent_for_insert = self.find_parent_for_insert(key)
        parent_for_insert_key = parent_for_insert.get_key()

        if parent_for_insert_key > key:
            parent_for_insert.set_left(leaf_for_insert)
        if parent_for_insert_key < key:
            parent_for_insert.set_right(leaf_for_insert)

        return parent_for_insert

    def delete(self, node: AVLNode):
        """Deletes node from the dictionary

        @type node: AVLNode
        @pre: node is a real pointer to a node in self
        @rtype: int
        @returns: the number of rebalancing operation due to AVL rebalancing

        @complexity: The operation has a time complexity of O(log n) in the worst case, where n is the number of nodes
            in the AVL tree. 
        """
        node_to_fix_from = None

        if node.is_leaf() or node.has_one_child():
            node_to_fix_from = self.physical_delete(node)
        else:
            successor = self.find_successor_for_node_with_two_childs(node)
            node_to_fix_from = self.physical_delete(successor)
            self.replace_node_in_tree(node, successor)

            # In case we deleted the node we are trying to fix from
            if node is node_to_fix_from:
                node_to_fix_from = successor

        return self.fix_tree(node_to_fix_from)

    def physical_delete(self, node: AVLNode):
        """Physically deletes a node with up to one child (virtual node is not considered a child)
        from the AVL tree, by adjusting the tree structure.

        The function handles the physical deletion of a node based on its type:
        - If node is not the root of self, its child is assigned as the new child of its parent.
        - If node is the root of self, if node has a child, node's child will become the new root. 
        otherwise, we will point root to None.

        @type node: AVLNode
        @param node: node to be physically deleted
        @rtype: AVLNode or None
        @returns: The parent node from which tree balancing should start after the physical deletion.
        """
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

    def replace_node_in_tree(self, old_node: AVLNode, new_node: AVLNode):
        """Replaces a node in the AVL tree with a new node.

        The function replaces the old node with the new node while preserving the old node's left and right children.
        If the old node is the root, the new node becomes the root.

        @type old_node: AVLNode
        @param old_node: The node to be replaced.
        @type new_node: AVLNode
        @param new_node: The new node to replace the old node.
        """
        if self.root == old_node:
            self.set_root(new_node)

        old_node_parent = old_node.get_parent()
        new_node.set_left(old_node.get_left())
        new_node.set_right(old_node.get_right())

        if old_node_parent != None:
            old_node_parent.update_parents_child(old_node, new_node)

    def avl_to_array(self):
        """Returns an array representing dictionary 

        @rtype: list
        @returns: a sorted list according to key of touples (key, value) representing the data structure

        @complexity: The operation has a time complexity of O(n) in the worst case, where n is the number of nodes
        in the AVL tree. 
        """
        array = []
        self.avl_to_array_rec(self.root, array)
        return array

    def avl_to_array_rec(self, node, array):
        """Recursively converts an AVL tree into an array representation.

        The function traverses the AVL tree in an in-order fashion (left-subtree, root, right-subtree)
        and appends each node's key-value pair to the provided array.

        @type node: AVLNode or None
        @param node: The current node being processed.
        @type array: list
        @param array: The array to store the key-value pairs of the AVL tree.

        @complexity: The time complexity of this function is O(n), where 'n' is the number of nodes in the AVL tree.
        """
        if AVLNode.is_empty_node(node):
            return
        self.avl_to_array_rec(node.get_left(), array)
        array.append((node.get_key(), node.get_value()))
        self.avl_to_array_rec(node.get_right(), array)

    def size(self):
        """Returns the number of items in dictionary 

        @rtype: int
        @returns: the number of items in dictionary 
        """
        return self.get_root().get_size()

    def split(self, node: AVLNode):
        """Splits the dictionary at a given node

        @type node: AVLNode
        @pre: node is in self
        @param node: The intended node in the dictionary according to whom we split
        @rtype: list
        @returns: a list [left, right], where left is an AVLTree representing the keys in the 
        dictionary smaller than node.key, right is an AVLTree representing the keys in the 
        dictionary larger than node.key.
        @complexity: The time complexity of fixing the tree is O(log n), where n is the number of nodes in the AVL tree.
        """
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

    def join_tree_with_array_of_node_tuples(self, array_of_node_tuples):
        """Joins the current AVLTree with other AVLTrees based on the given array of node tuples.

        @type array_of_node_tuples: List[Tuple(AVLNode, AVLNode)]
        @param array_of_node_tuples: The array of node tuples containing connecting nodes and trees to join.
        @complexity: O(log n), where n is the number of nodes in the AVL tree.
        """
        for mid_node, other_node in array_of_node_tuples:
            other_tree = self.create_tree(other_node)
            self.join(other_tree, mid_node.get_key(), mid_node.get_value())

    def join(self, tree, key, val):
        """Joins self with key and another AVLTree

        @type tree: AVLTree 
        @param tree: a dictionary to be joined with self
        @type key: int 
        @param key: The key separting self with tree
        @type val: any 
        @param val: The value attached to key
        @pre: all keys in self are smaller than key and all keys in tree are larger than key,
        or the other way around.
        @rtype: int
        @returns: the absolute value of the difference between the height of the AVL trees joined plus 1
        @complexity: The time complexity of fixing the tree is O(log n), where n is the number of nodes in the AVL tree.
        """
        if AVLNode.is_empty_node(self.root) and AVLNode.is_empty_node(tree.root):
            self.set_root(AVLNode.create_leaf_with_virtual_nodes(key, val))
            return 1

        # Find the tree with the lower values (t1) and the tree with the higher values (t2)
        if not AVLNode.is_empty_node(self.root):
            if tree.root == None:
                tree.root = AVLNode(None, None)
            if self.root.get_key() < key:
                t1 = self
                t2 = tree
            else:
                t1 = tree
                t2 = self
        else:
            if self.root == None:
                self.root = AVLNode(None, None)
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

    def create_tree(self, root):
        """Creates a new AVLTree with the given root node.

        @type root: AVLNode
        @param root: The root node of the AVLTree to be created.
        @rtype: AVLTree
        @returns: The created AVLTree with the given root node.
        """
        tree = AVLTree()
        tree.set_root(root)
        return tree

    def join_trees_with_equal_heights(self, t1, t2, x):
        """Joins two AVL trees with equal heights and attaches a new root node x between them.

        @type t1: AVLTree
        @param t1: The left subtree join.
        @type t2: AVLTree
        @param t2: The right subtree to join.
        @type x: AVLNode
        @param x: The new root node to attach between the two AVL trees.
        """
        x.set_left(t1.root)
        x.set_right(t2.root)
        self.root = x

    def join_trees_left_tree_is_smaller(self, t1, t2, x):
        """Joins two AVL trees where the left tree is smaller in height than the right tree.

        @type t1: AVLTree
        @param t1: The AVLTree with the smaller height.
        @type t2: AVLTree
        @param t2: The AVLTree with the greater height.
        @type x: AVLNode
        @param x: The new root node to attach between the two subtrees.
        @complexity: O(log n), where n is the number of nodes in the AVL tree.
        """
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

    def join_trees_right_tree_is_smaller(self, t1, t2, x):
        """Joins two AVL trees where the right tree is smaller in height than the left tree.

        @type t1: AVLTree
        @param t1: The AVLTree with the greater height.
        @type t2: AVLTree
        @param t2: The AVLTree with the smaller height.
        @type x: AVLNode
        @param x: The new root node to attach between the two subtrees.
        @complexity: O(log n), where n is the number of nodes in the AVL tree
        """
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

    def rank(self, node):
        """Compute the rank of node in the self

        @type node: AVLNode
        @pre: node is in self
        @param node: a node in the dictionary which we want to compute its rank
        @rtype: int
        @returns: the rank of node in self
        @complexity: O(log n), where n is the number of nodes in the AVL tree
        """
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

    def select(self, i):
        """Finds the i'th smallest item (according to keys) in self

        @type i: int
        @pre: 1 <= i <= self.size()
        @param i: the rank to be selected in self
        @rtype: AVLNode
        @returns: the item of rank i in self
        @complexity: O(log n), where n is the number of nodes in the AVL tree.
        """
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

    def get_root(self):
        """Returns the root of the tree representing the dictionary

        @rtype: AVLNode or None
        @returns: the root, None if the dictionary is empty
        """
        return self.root
