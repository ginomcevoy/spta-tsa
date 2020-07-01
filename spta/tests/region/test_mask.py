'''
Unit tests for spta.region.mask module.
'''
import unittest

from spta.region import Point
from spta.region.mask import MaskRegionCrisp, MaskRegionFuzzy
from spta.tests.stub import stub_mask


class TestMaskRegionCrisp(unittest.TestCase):
    '''
    Unit tests for mask.MaskRegionCrisp class.
    '''

    def setUp(self):
        self.mask_np = stub_mask.crisp_membership_stub()

    def test_find_memberships_empty(self):

        # given a cluster mask and no points
        cluster_index = 1
        mask = MaskRegionCrisp(self.mask_np, cluster_index)
        points = []

        # when asking for memberships
        result = mask.find_memberships(points)

        # then result is empty
        expected = []
        self.assertEqual(result, expected)

    def test_find_memberships_single(self):

        # given a cluster mask and one point
        cluster_index = 1
        mask = MaskRegionCrisp(self.mask_np, cluster_index)
        points = [Point(0, 0)]

        # when asking for memberships
        result = mask.find_memberships(points)

        # then result is membership of point
        expected = [0]
        self.assertEqual(result, expected)

    def test_find_memberships_three(self):

        # given a cluster mask and three points
        cluster_index = 1
        mask = MaskRegionCrisp(self.mask_np, cluster_index)
        points = [Point(0, 0), Point(2, 4), Point(1, 1)]

        # when asking for memberships
        result = mask.find_memberships(points)

        # then result is memberships of points
        expected = [0, 2, 1]
        self.assertEqual(result, expected)


class TestMaskRegionFuzzy(unittest.TestCase):
    '''
    Unit tests for mask.MaskRegionFuzzy class.
    '''

    def setUp(self):
        self.mask_np = stub_mask.mask_fuzzy_np_stub()
        self.membership_np = stub_mask.fuzzy_membership_stub()

    def test_constructor_good(self):
        # given
        cluster_index = 1
        threshold = 0.5

        # when
        mask = MaskRegionFuzzy(self.mask_np, cluster_index, threshold)

        # then
        self.assertTrue(mask.numpy_dataset is self.mask_np)
        self.assertTrue(mask.cluster_index is cluster_index)
        self.assertTrue(mask.threshold is threshold)

    def test_constructor_bad_cluster_index(self):
        # given an index too big for given mask
        cluster_index = 2
        threshold = 0

        # then assertion error
        with self.assertRaises(AssertionError):
            MaskRegionFuzzy(self.mask_np, cluster_index, threshold)

    def test_constructor_bad_threshold(self):
        # given a threshold too high
        cluster_index = 1
        threshold = 1.1

        # then assertion error
        with self.assertRaises(AssertionError):
            MaskRegionFuzzy(self.mask_np, cluster_index, threshold)

    def test_is_member_own_medoid(self):
        # given a fuzzy cluster
        cluster_index = 1
        threshold = 0.5

        # when
        mask = MaskRegionFuzzy(self.mask_np, cluster_index, threshold)

        # when asking for its medoid
        medoid = Point(0, 0)

        # then it's member
        self.assertTrue(mask.is_member(medoid))

    def test_is_member_other_medoid(self):
        # given a fuzzy cluster
        cluster_index = 0
        threshold = 0.5

        # when
        mask = MaskRegionFuzzy(self.mask_np, cluster_index, threshold)

        # when asking for other medoid
        medoid = Point(0, 0)

        # then it's not member
        self.assertFalse(mask.is_member(medoid))

    def test_is_member_no_threshold_best(self):
        # given a fuzzy cluster with no threshold
        cluster_index = 1
        threshold = 0
        mask = MaskRegionFuzzy(self.mask_np, cluster_index, threshold)

        # when asking for a point where uij is the highest for this cluster
        point_in_cluster = Point(0, 1)

        # then it's member
        self.assertTrue(mask.is_member(point_in_cluster))

    def test_is_member_no_threshold_not_best(self):
        # given a fuzzy cluster with no threshold
        cluster_index = 1
        threshold = 0
        mask = MaskRegionFuzzy(self.mask_np, cluster_index, threshold)

        # when asking for a point where uij is the not the highest for this cluster
        point_not_in_cluster = Point(3, 4)

        # then it's not a member
        self.assertFalse(mask.is_member(point_not_in_cluster))

    def test_is_member_by_threshold(self):
        # given a fuzzy cluster
        cluster_index = 1
        threshold = 0.05
        mask = MaskRegionFuzzy(self.mask_np, cluster_index, threshold)

        # when asking for a point that can be considered a member only due to threshold
        # mask_np[:, 0, 4] = [0.52156374, 0.47843626]
        point_in_cluster_by_threshold = Point(0, 4)

        # then it's considered a member
        self.assertTrue(mask.is_member(point_in_cluster_by_threshold))

    def test_is_member_threshold_too_low(self):
        # given a fuzzy cluster
        cluster_index = 1
        threshold = 0.03
        mask = MaskRegionFuzzy(self.mask_np, cluster_index, threshold)

        # when asking for a point that cannot be considered a member, threshold too low
        # mask_np[:, 0, 4] = [0.52156374, 0.47843626]
        point_in_cluster_by_threshold = Point(0, 4)

        # then it's not considered a member
        self.assertFalse(mask.is_member(point_in_cluster_by_threshold))

    def test_iterator_no_threshold(self):
        # given a fuzzy cluster with no threshold
        cluster_index = 1
        threshold = 0
        mask = MaskRegionFuzzy(self.mask_np, cluster_index, threshold)

        # when iterating over all points
        members = [point for point in mask]

        # then only members with the highest membership value are iterated
        self.assertEquals(len(members), 9)

    def test_iterator_with_threshold(self):
        # given a fuzzy cluster
        cluster_index = 1
        threshold = 0.10
        mask = MaskRegionFuzzy(self.mask_np, cluster_index, threshold)

        # when iterating over all points
        members = [point for point in mask]

        # then more members are iterated because of the threshold
        # for two clusters, the membership index should be at least 0.45 (0.55 - 0.45 = 0.10)
        self.assertEquals(len(members), 15)

    def test_from_uij_and_region(self):
        # given the uij 2-d matrix with shape (20, 2) for 20 points and 2 clusters
        uij = self.membership_np

        # when using class method
        mask = MaskRegionFuzzy.from_uij_and_region(uij, x_len=4, y_len=5, cluster_index=1,
                                                   threshold=0.10)

        # then a valid mask is created
        self.assertIsNotNone(mask)

        # then we keep some properties already tested
        self.assertTrue(mask.is_member(Point(0, 0)))    # medoid
        self.assertTrue(mask.is_member(Point(0, 4)))    # highest membership
        self.assertTrue(mask.is_member(Point(0, 1)))    # by threshold

        self.assertFalse(mask.is_member(Point(3, 3)))   # other medoid
        self.assertFalse(mask.is_member(Point(3, 4)))   # too far from cluster

        # test iterator
        members = [point for point in mask]
        self.assertEquals(len(members), 15)

        # test it again to ensure it can be iterated many times
        members = [point for point in mask]
        self.assertEquals(len(members), 15)

    def test_varying_threshold(self):

        # given a fuzzy cluster with initial threshold
        cluster_index = 1
        threshold = 0
        mask = MaskRegionFuzzy(self.mask_np, cluster_index, threshold)

        # when iterating over all points, varying threshold and iterating again, then reverting
        members_1 = [point for point in mask]

        mask.threshold = 0.10
        members_2 = [point for point in mask]

        mask.threshold = 0
        members_3 = [point for point in mask]

        # then all iterations work as expected
        self.assertEquals(len(members_1), 9)
        self.assertEquals(len(members_2), 15)
        self.assertEquals(len(members_3), 9)



