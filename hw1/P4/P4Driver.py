from mrjob.job import MRJob
import math
import sys
import P4

if __name__ == '__main__':
	# want to stop running if there are no changes made (set converged)
	converged = False
	
	init_input = open('graph.txt', 'r')

	while(not converged):
		# setup and run MRJob instance
		mrJobInstance = P4.mrGraphAlg()
		mrJobInstance.stdin = init_input
		runner = mrJobInstance.make_runner()
		runner.run()
		# pull out the counts from the MRJob 
		counts = runner.counters()
	        # if the counter is 0, no changes, graph converged
		if (counts[0]['graph']['node'] == 0):
			converged = True
		# save all of the new graph information for the next run, or as output
		f_out = open('out.txt','w')
		for line in runner.stream_output():
			f_out.write(line)
		f_out.close()
		init_input.close()
		# reopen out.txt and use that for next iteration
		init_input = open('out.txt','r')
