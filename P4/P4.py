from mrjob.job import MRJob
import math
import sys
from mrjob.protocol import JSONProtocol

class mrGraphAlg(MRJob):
	INPUT_PROTOCOL = JSONProtocol

	

	def mapper(self, node, neighbors):
		yield node,("graph_type",neighbors)

		dist = neighbors.pop(0)
		for (n, d) in neighbors:
			yield n, ("distance_type",d + dist)

	def reducer(self, node, values):
		updated_neighbors = []
		min_dist = 999
		self.increment_counter("graph","node",0) #if counter is never hit later, it won't exist	
		for(itemtype,itemvalue) in values:
			if (itemtype == "graph_type"):
				updated_neighbors = itemvalue
			else:
				assert (itemtype == "distance_type")
				min_dist = min(min_dist,itemvalue)

		curr_dist = updated_neighbors[0]
		if(min_dist < curr_dist):
			updated_neighbors[0] = min_dist
			self.increment_counter("graph","node",1)
		yield node, updated_neighbors

if __name__ == '__main__':
	mrGraphAlg.run()
