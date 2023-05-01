

from AVLTree import AVLTree
from AVLTree import AVLNode


def test_rotate_right_insert():
    tree = AVLTree()
    tree.insert(12, 0)
    tree.insert(8, 0)
    tree.insert(15, 0)
    tree.insert(6, 0)
    tree.insert(10, 0)
    tree.insert(14, 0)
    tree.insert(24, 0)
    tree.insert(11, 0)
    tree.insert(13, 0)
    tree.insert(20, 0)
    tree.insert(29, 0)
    tree.insert(19, 0)
    tree.insert(18, 0)
    print(tree)


def test_rotate_left_insert():
    tree = AVLTree()
    tree.insert(12, 0)
    tree.insert(8, 0)
    tree.insert(15, 0)
    tree.insert(6, 0)
    tree.insert(10, 0)
    tree.insert(14, 0)
    tree.insert(24, 0)
    tree.insert(11, 0)
    tree.insert(29, 0)
    tree.insert(5, 0)
    tree.insert(9.5, 0)
    tree.insert(9, 0)
    tree.insert(8.5, 0)
    print(tree)


def test_rotate_left_right_insert():
    tree = AVLTree()
    tree.insert(15, 0)
    tree.insert(10, 0)
    tree.insert(22, 0)
    tree.insert(4, 0)
    tree.insert(11, 0)
    tree.insert(20, 0)
    tree.insert(24, 0)
    tree.insert(2, 0)
    tree.insert(7, 0)
    tree.insert(12, 0)
    tree.insert(18, 0)
    tree.insert(1, 0)
    tree.insert(6, 0)
    tree.insert(8, 0)
    tree.insert(5, 0)
    print(tree)


def test_rotate_right_left_insert():
    tree = AVLTree()
    tree.insert(15, 0)
    tree.insert(10, 0)
    tree.insert(20, 0)
    tree.insert(4, 0)
    tree.insert(11, 0)
    tree.insert(18, 0)
    tree.insert(22, 0)
    tree.insert(2, 0)
    tree.insert(17, 0)
    tree.insert(21, 0)
    tree.insert(24, 0)
    tree.insert(20.5, 0)
    tree.insert(21.5, 0)
    tree.insert(25, 0)
    tree.insert(20.25, 0)
    print(tree)


def test_leaf_delete():
    tree = AVLTree()
    tree.insert(15, 0)
    tree.insert(10, 0)
    tree.insert(20, 0)
    tree.insert(4, 0)
    tree.insert(11, 0)
    tree.insert(18, 0)
    tree.insert(22, 0)
    tree.insert(2, 0)
    tree.insert(17, 0)
    tree.insert(21, 0)
    tree.insert(24, 0)
    print("BEFORE:--------------------------------------")
    print(tree)
    print("---------------------------------------------")
    node_to_delete = tree.search(2)
    tree.delete(node_to_delete)
    print("AFTER:--------------------------------------")
    print(tree)
    print("---------------------------------------------")


def test_node_with_one_child_delete():
    tree = AVLTree()
    tree.insert(15, 0)
    tree.insert(10, 0)
    tree.insert(20, 0)
    tree.insert(4, 0)
    tree.insert(11, 0)
    tree.insert(18, 0)
    tree.insert(22, 0)
    tree.insert(2, 0)
    tree.insert(17, 0)
    tree.insert(21, 0)
    tree.insert(24, 0)
    print("BEFORE:--------------------------------------")
    print(tree)
    print("---------------------------------------------")
    node_to_delete = tree.search(18)
    tree.delete(node_to_delete)
    print("AFTER:--------------------------------------")
    print(tree)
    print("---------------------------------------------")


def test_node_with_two_childs_delete():
    tree = AVLTree()
    tree.insert(15, 0)
    tree.insert(10, 0)
    tree.insert(20, 0)
    tree.insert(4, 0)
    tree.insert(11, 0)
    tree.insert(18, 0)
    tree.insert(22, 0)
    tree.insert(2, 0)
    tree.insert(17, 0)
    tree.insert(21, 0)
    tree.insert(24, 0)
    tree.insert(19, 0)
    print("BEFORE:--------------------------------------")
    print(tree)
    print("---------------------------------------------")
    node_to_delete = tree.search(18)
    tree.delete(node_to_delete)
    print("AFTER:--------------------------------------")
    print(tree)
    print("---------------------------------------------")


def test_rotate_right_delete():
    tree = AVLTree()
    tree.insert(8, 0)
    tree.insert(5, 0)
    tree.insert(10, 0)
    tree.insert(4, 0)
    tree.insert(6, 0)
    print("BEFORE:--------------------------------------")
    print(tree)
    print("---------------------------------------------")
    node_to_delete = tree.search(10)
    tree.delete(node_to_delete)
    print("AFTER:--------------------------------------")
    print(tree)
    print("---------------------------------------------")


def test_rotate_right_then_left_right_delete():
    tree = AVLTree()
    tree.insert(15, 0)
    tree.insert(8, 0)
    tree.insert(22, 0)
    tree.insert(4, 0)
    tree.insert(11, 0)
    tree.insert(20, 0)
    tree.insert(24, 0)
    tree.insert(2, 0)
    tree.insert(9, 0)
    tree.insert(12, 0)
    tree.insert(18, 0)
    tree.insert(13, 0)
    print("BEFORE:--------------------------------------")
    print(tree)
    print("---------------------------------------------")
    node_to_delete = tree.search(24)
    num_of_rotations = tree.delete(node_to_delete)
    print("AFTER:--------------------------------------")
    print(tree)
    print(num_of_rotations)
    print("---------------------------------------------")


def test_avl_to_array():
    tree = AVLTree()
    tree.insert(15, 0)
    tree.insert(8, 0)
    tree.insert(22, 0)
    tree.insert(4, 0)
    tree.insert(11, 0)
    tree.insert(20, 0)
    tree.insert(24, 0)
    tree.insert(2, 0)
    tree.insert(9, 0)
    tree.insert(12, 0)
    tree.insert(18, 0)
    tree.insert(13, 0)
    print("BEFORE:--------------------------------------")
    print(tree)
    print("---------------------------------------------")
    print(tree.avl_to_array())


def test_size():
    tree = AVLTree()
    tree.insert(15, 0)
    tree.insert(8, 0)
    tree.insert(22, 0)
    tree.insert(4, 0)
    tree.insert(11, 0)
    tree.insert(20, 0)
    tree.insert(24, 0)
    tree.insert(2, 0)
    tree.insert(9, 0)
    tree.insert(12, 0)
    tree.insert(18, 0)
    tree.insert(13, 0)
    print("BEFORE:--------------------------------------")
    print(tree)
    print("---------------------------------------------")
    print(tree.size())


def test_join_equal_heights_trees():
    t1 = AVLTree()
    t2 = AVLTree()
    x = (10, 0)
    t1.insert(5, 0)
    t1.insert(4, 0)
    t1.insert(6, 0)
    t2.insert(15, 0)
    t2.insert(14, 0)
    t2.insert(16, 0)
    t1.join(t2, x[0], x[1])
    print(t1)


def test_join_left_tree_is_taller_by_1():
    t1 = AVLTree()
    t2 = AVLTree()
    x = (10, 0)
    t1.insert(5, 0)
    t1.insert(4, 0)
    t1.insert(6, 0)
    t1.insert(3, 0)
    t1.insert(7, 0)
    t2.insert(15, 0)
    t2.insert(14, 0)
    t2.insert(16, 0)
    print(t1)
    print(t2)
    t1.join(t2, x[0], x[1])
    print(t1)


def test_join_right_tree_is_taller_by_1():
    t1 = AVLTree()
    t2 = AVLTree()
    x = (10, 0)
    t1.insert(5, 0)
    t1.insert(4, 0)
    t1.insert(6, 0)
    t2.insert(15, 0)
    t2.insert(14, 0)
    t2.insert(16, 0)
    t2.insert(13, 0)
    t2.insert(17, 0)
    print(t1)
    print(t2)
    t1.join(t2, x[0], x[1])
    print(t1)


def test_split():
    tree = AVLTree()
    tree.insert(15, 0)
    tree.insert(8, 0)
    tree.insert(22, 0)
    tree.insert(4, 0)
    tree.insert(11, 0)
    tree.insert(20, 0)
    tree.insert(24, 0)
    tree.insert(2, 0)
    tree.insert(9, 0)
    tree.insert(12, 0)
    tree.insert(18, 0)
    tree.insert(13, 0)
    print(tree)
    t1, t2 = tree.split(tree.search(12))
    print('------------------------------------')
    print(t1)
    print('------------------------------------')
    print(t2)
    print('------------------------------------')


# test_rotate_right_insert()
# test_rotate_left_insert()
# test_rotate_left_right_insert()
# test_rotate_right_left_insert()
# test_leaf_delete()
# test_node_with_one_child_delete()
# test_node_with_two_childs_delete()
# test_rotate_right_delete()
# test_rotate_right_then_left_right_delete()
# test_avl_to_array()
# test_size()
# test_join_equal_heights_trees()
# test_join_left_tree_is_taller_by_1()
# test_join_right_tree_is_taller_by_1()
test_split()