from matplotlib import pyplot as plt
import d2_neg.plot as p2
import d3_neg.plot as p3

fig, (ax2, ax3) = plt.subplots(1,2, sharey=True)
p2.plot_main(ax2, 0.6, 0.6)
p3.plot_main(ax3, 0.6, 0.6)
fig.savefig('main_super.pdf')


