from mrjob.job import MRJob
import math
import sys
from mrjob.protocol import JSONProtocol

class mrGraphAlg(MRJob):
        # set up IO to handle JSON input
        INPUT_PROTOCOL = JSONProtocol

        def mapper(self, node, neighbors):
                # yield the graph structure so we can update the entire graph in reducer
                yield node,("graph_type",neighbors)
                dist = neighbors.pop(0)
                for (n, d) in neighbors:
                        # update distance for next node. 
			#dist to current node + dist to neighbor
                        yield n, ("distance_type",d + dist)

        def reducer(self, node, values):
                updated_neighbors = []
                min_dist = 999
                # if counter is never hit later, it won't exist
		# initialize it for every run
                self.increment_counter("graph","node",0)	
                for(itemtype,itemvalue) in values:
                        # if the received item is the graph, store it all
                        if (itemtype == "graph_type"):
                                updated_neighbors = itemvalue
                        else:
                                # we received updated dist 
				# pick b/t the min dist and the received dist
                                assert (itemtype == "distance_type")
                                min_dist = min(min_dist,itemvalue)

                curr_dist = updated_neighbors[0]
                if(min_dist < curr_dist):
                        # if we update the dist, then increment counter.
                        # we will converge on 0 if no distances are updated
                        updated_neighbors[0] = min_dist
                        self.increment_counter("graph","node",1)
                yield node, updated_neighbors

if __name__ == '__main__':
        mrGraphAlg.run()
