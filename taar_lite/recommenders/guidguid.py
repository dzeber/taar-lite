# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at http://mozilla.org/MPL/2.0/.
from .treatments import BaseTreatment


class GuidGuidCoinstallRecommender:
    """ A recommender class that returns top N addons based on a
    passed addon identifier.
    Accepts:
        - raw_coinstall_dict: a dict containing coinstalled addons
        - treatments: list of treatments that transform the original coinstall dict, and will
          be applied in the order supplied
        - treatment_kwargs: additional named args that will be passed to the treatments
        - tie_breaker_dict: a dict of addon rankings used to break ties when ordering recommendations
        - apply_treatment_on_init: should the treated dataset be constructed immediately
          on initialization? Default True.
        - validate_raw_coinstall_dict: verify that raw_coinstall_dict represents
          a symmetric matrix? Default True.

    Provides a recommend method to then return recommendations for a supplied addon.
    Can also return the complete recommendation graph.
    """

    def __init__(
            self,
            raw_coinstall_dict,
            treatments,
            treatment_kwargs=None,
            tie_breaker_dict=None,
            apply_treatment_on_init=True,
            validate_raw_coinstall_dict=True):

        for treatment in treatments:
            assert isinstance(treatment, BaseTreatment)

        if validate_raw_coinstall_dict:
            self.validate_coinstall_dict(raw_coinstall_dict)

        if not tie_breaker_dict:
            tie_breaker_dict = dict()

        if not treatment_kwargs:
            treatment_kwargs = dict()

        self._raw_coinstall_graph = raw_coinstall_dict
        self._tie_breaker_dict = tie_breaker_dict
        self._treatment_kwargs = treatment_kwargs
        self._treatments = treatments
        self._treated_graph = dict()

        if apply_treatment_on_init:
            self.build_treatment_graph()

    @staticmethod
    def validate_coinstall_dict(coinstalls):
        """Verifies that a coinstallation dict represents a symmetric matrix.

        This means that, for any nested keys (i, j) in the coinstall dict,

            coinstalls[i][j] == coinstalls[j][i],

        and the dict contains both entries.
        """
        for outer_guid in coinstalls:
            for inner_guid in coinstalls[outer_guid]:
                assert inner_guid in coinstalls
                assert outer_guid in coinstalls[inner_guid]
                assert coinstalls[outer_guid][inner_guid] == coinstalls[inner_guid][outer_guid]


    @property
    def raw_coinstall_graph(self):
        """Returns a dictionary with guid keys and a coinstall set values.

        Something like this, but with much longer values.

            {
                'guid_a': {'guid_b': 10, 'guid_c': 13},
                'guid_b': {'guid_a': 10, 'guid_c': 4},
                'guid_c': {'guid_a': 13, 'guid_b': 4}
            }

        It must be symmetric.
        """
        return self._raw_coinstall_graph

    @property
    def tie_breaker_dict(self):
        """Returns a dict used for tie-breaking.

        The values are used to order items in the case where the treated
        values are the same.

        The keys will typically match the keys of the raw_coinstall_graph, but
        this is not validated, and missing keys are assinged a ranking value of 0
        in self._build_sorted_result_list.
        """
        return self._tie_breaker_dict

    @property
    def treated_graph(self):
        """Returns the recommentaion graph.

        Recommendation graph is in the same format as the coinstall graph but the
        numerical values are the weightings as a result of the treatment.
        """
        return self._treated_graph

    @property
    def treatments(self):
        """Return the list of treatments."""
        return self._treatments

    @property
    def treatment_kwargs(self):
        return self._treatment_kwargs

    def get_recommendation_graph(self, limit):
        """The recommendation graph is the full output for all addons"""
        rec_graph = {}
        for guid in self.raw_coinstall_graph:
            rec_graph[guid] = self.recommend(guid, limit)
        return rec_graph

    def build_treatment_graph(self):
        """Does the work to compute and then set the recommendation graph.
        Sub classes may wish to override if more complex computation is required.
        """
        new_graph = self.raw_coinstall_graph
        for treatment in self.treatments:
            new_graph = treatment.treat(new_graph, **self.treatment_kwargs)
        self._treated_graph = new_graph

    def recommend(self, for_guid, limit):
        """Returns a list of sorted recommendations of length 0 - limit for supplied guid.

        Result list is a list of tuples with the lex ranking string. e.g.
            [
                ('guid_a', '000003.000002.0001000'),
                ('guid_c', '000001.000002.0001000'),
                ('guid_b', '000001.000002.0000010'),
            ]

        """
        if for_guid not in self.treated_graph:
            return []
        raw_recommendations = self.treated_graph[for_guid]
        result_list = self._build_sorted_result_list(raw_recommendations)
        return result_list[:limit]

    def _build_sorted_result_list(self, unranked_recommendations):
        """Takes a dictionary with a format matching the values in the coinstall_dict
        and return a sorted list of results

            In: {'guid_b': 10, 'guid_c': 13}
            Out:
                [
                    ('guid_c', '000013.000000.0001000'),
                    ('guid_b', '000010.000000.0000010'),
                ]

        """
        # Augment the result_dict with the installation counts
        # and then we can sort using lexical sorting of strings.
        # The idea here is to get something in the form of
        #    0000.0000.0000
        # The computed weight takes the first and second segments of
        # integers.  The third segment is the installation count of
        # the addon but is zero padded.

        result_dict = {}
        for k, v in unranked_recommendations.items():
            lex_value = "{0:020.10f}.{1:010d}".format(v, self.tie_breaker_dict.get(k, 0))
            result_dict[k] = lex_value
        # Sort the result dictionary in descending order by weight
        result_list = sorted(result_dict.items(), key=lambda x: x[1], reverse=True)
        return result_list
