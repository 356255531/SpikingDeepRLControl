import numpy as np
import nengo

from nengo_gui.components.component import Component
from nengo_gui.components.spa_plot import SpaPlot


class SpaSimilarity(SpaPlot):
    """Line graph showing semantic pointer decoded values over time"""

    config_defaults = dict(max_value=1.5, min_value=-1.5,
                           show_pairs=False,
                           **Component.config_defaults)

    def __init__(self, obj, **kwargs):
        super(SpaSimilarity, self).__init__(obj, **kwargs)

        self.old_vocab_length = len(self.vocab_out.keys)
        self.old_pairs_length = 0
        self.labels = self.vocab_out.keys
        self.previous_pairs = False

        # Nengo objects for data collection
        self.node = None
        self.conn = None

    def add_nengo_objects(self, page):
        with page.model:
            output = self.obj.outputs[self.target][0]
            self.node = nengo.Node(self.gather_data,
                                   size_in=self.vocab_out.dimensions)
            self.conn = nengo.Connection(output, self.node, synapse=0.01)

    def remove_nengo_objects(self, page):
        """Undo the changes made by add_nengo_objects."""
        page.model.connections.remove(self.conn)
        page.model.nodes.remove(self.node)

    def gather_data(self, t, x):
        vocab = self.vocab_out

        if self.old_vocab_length != len(vocab.keys):
            self.update_legend(vocab)

        # get the similarity and send it
        key_similarity = np.dot(vocab.vectors, x)
        simi_list = ['{:.2f}'.format(simi) for simi in key_similarity]

        if self.config.show_pairs:

            # briefly there can be no pairs, so catch the error
            try:
                pair_similarity = np.dot(vocab.vector_pairs, x)
                simi_list += ['{:.2f}'.format(simi) for simi in pair_similarity]
            except TypeError:
                pass

        if(simi_list != []):
            self.data.append(  '["data_msg", %g, %s]'
                             %( t, ",".join(simi_list) )  )

    def update_legend(self, vocab):
        # pass all the missing keys
        legend_update = []
        legend_update += (vocab.keys[self.old_vocab_length:])
        self.old_vocab_length = len(vocab.keys)
        # and all the missing pairs if we're showing pairs
        if self.config.show_pairs:
            # briefly there can be no pairs, so catch the error
            try:
                legend_update += vocab.key_pairs[self.old_pairs_length:]
                self.old_pairs_length = len(vocab.key_pairs)
            except TypeError:
                pass

        self.data.append('["update_legend", "%s"]'
                         %('","'.join(legend_update)))

    def javascript(self):
        """Generate the javascript that will create the client-side object"""
        info = dict(uid=id(self), label=self.label, n_lines=len(self.labels),
                    synapse=0, pointer_labels=self.labels)
        json = self.javascript_config(info)
        return 'new Nengo.SpaSimilarity(main, sim, %s);' % json

    def message(self, msg):
        """Message receive function for show_pairs toggling and reset"""
        vocab = self.vocab_out
        # Send the new labels
        if self.config.show_pairs:
            vocab.include_pairs = True
            self.data.append(
                '["reset_legend_and_data", "%s"]' % (
                    '","'.join(vocab.keys + vocab.key_pairs)))
            # if we're starting to show pairs, track pair length
            self.old_pairs_length = len(vocab.key_pairs)
        else:
            vocab.include_pairs = False
            self.data.append('["reset_legend_and_data", "%s"]'
                             % ('","'.join(vocab.keys)))
