"""Actual CUDA submission parity for MC -> level-pool -> MC dependencies."""
import unittest

import cupy as cp

from gpuwm.hydro.reservoirs import Reservoirs, PARAMETERS, CONTROLS, SOURCE_PIN
from gpuwm.hydro.troute_mc import MuskingumCunge, ReservoirLevel


class RoutingGraphTests(unittest.TestCase):
    def router(self):
        r = {key: cp.zeros(1, cp.float32) for key in (
            'previous_inflow', 'inflow', 'lateral', 'height', 'initial_height',
            'levelpool_release', 'levelpool_height')}
        r['parameters'] = cp.asarray(
            [[1], [1], [10], [.6], [1], [1], [0], [.6], [1]], cp.float32)
        r.update({key: cp.zeros(1, cp.int32) for key in (
            'indices', 'dynamic_type', 'active_source', 'fault')})
        r['types'] = cp.ones(1, cp.int32)
        r['offsets'] = cp.zeros(2, cp.int64)
        r.update({key: cp.zeros(1, cp.float64) for key in (
            'clock', 'release', 'assimilated')})
        r['carriers'] = cp.zeros((3, 1), cp.float64)
        r['controls'] = cp.zeros((8, 1), cp.float64)
        r['series'] = cp.empty(0, cp.float64)
        r['times'] = cp.empty(0, cp.float64)
        reservoirs = Reservoirs(r, dict(parameter_columns=PARAMETERS,
            control_columns=CONTROLS, source_pin=SOURCE_PIN, validated_csr=True,
            network_sha256='submission-fixture', parameters_sha256='submission-fixture',
            series_sha256='empty', time_origin='fixture', reservoir_order_sha256='one'))
        router = MuskingumCunge(dt_seconds=300.,
            parameters=cp.asarray([[1000, 10, 20, 30, .03, .06, 1, .001]]*3, cp.float32),
            state=cp.asarray([[3, .25, .5], [0, 0, 0], [5, .5, .75]], cp.float32),
            lateral=cp.full(3, .2, cp.float32),
            upstream_offsets=cp.asarray([0, 0, 1, 2], cp.int32),
            upstream_indices=cp.asarray([0, 1], cp.int32),
            ordered_nodes=cp.asarray([0, 2], cp.int32), level_offsets=[0, 1, 1, 2],
            boundary_previous=cp.zeros(3, cp.float32), boundary_current=cp.zeros(3, cp.float32),
            reach_ids=cp.asarray([17, 23, 29], cp.int64), publish_channels=True,
            identity=dict(network_sha256='submission-fixture', parameters_sha256='fixture',
                          ordering_sha256='MC-reservoir-MC', time_origin='fixture'))
        hook = ReservoirLevel(router, reservoirs, rows=cp.zeros(1, cp.int32),
            node_indices=cp.ones(1, cp.int32), selection_sha256='row-zero',
            mapping_sha256='node-one', validated_mapping=True)
        router.bind_level_hooks({1: [hook]})
        return router, hook

    def words(self, router):
        return {key: value.get().tobytes() for key, value in router.arrays.items()}

    def test_capture_preserves_all_carriers_and_dynamic_inputs_bitwise(self):
        direct, _ = self.router()
        graph, hook = self.router()
        before = self.words(graph)
        graph.prepare_graph()
        self.assertEqual(before, self.words(graph), 'capture must not advance water')
        for step in range(31):
            for router in (direct, graph):
                router.arrays['lateral'].fill((step + 1) * .125)
                router.launch(300.)
            self.assertEqual(self.words(direct), self.words(graph), f'step {step}')
        self.assertEqual(float(graph.arrays['reservoir/clock'][0].get()), 31 * 300.)
        # During an outer forecast capture, expose the original ordered kernels
        # instead of attempting an unsupported nested executable graph launch.
        stream = cp.cuda.Stream(non_blocking=True)
        cp.cuda.get_current_stream().synchronize()
        with stream:
            stream.begin_capture()
            graph.launch(300., stream=stream)
            outer = stream.end_capture()
            outer.launch(stream=stream)
        completion = cp.cuda.Event(block=True, disable_timing=True)
        completion.record(stream)
        completion.synchronize()
        direct.launch(300.)
        self.assertEqual(self.words(direct), self.words(graph))
        graph.bind_level_hooks({1: [hook]})
        self.assertIsNone(graph._graph, 'new hook binding invalidates captured pointers')
        self.assertNotIn('submission', graph.restart_identity)
        graph.prepare_graph()
        with self.assertRaisesRegex(ValueError, 'cadence'):
            graph.launch(301.)


if __name__ == '__main__':
    unittest.main(verbosity=2)
