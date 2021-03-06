"""
Trains a model using one or more GPUs.
"""

from multiprocessing import Process
import caffe
import numpy as np
from pylab import zeros, arange, subplots, plt, savefig
import glob, os

dataset_percetage = 10
training_id = 'iMaterialistFashion_ImageNetInit_all1loss_dataset'+str(dataset_percetage) # name to save the training plots
solver_path = 'prototxt/solver_multiGPU.prototxt' # solver proto definition
# snapshot = '../../../hd/datasets/instaFashion/models/CNNRegression/.caffemodel' # snapshot to restore (only weights initialzation)
snapshot = '../../../hd/datasets/SocialMedia/models/pretrained/bvlc_googlenet.caffemodel'
# snapshot = 0
gpus = [1] # list of device ids # last GPU requires por mem (9000-5000)
timing = False # show timing info for compute and communications
plotting = True # plot loss
test_interval = int(20000 * (float(dataset_percetage)/100)) # 5000 do validation each this iterations #5000
test_iters = 150 # number of validation iterations #20


def train(solver_path,  snapshot,  gpus):
    # NCCL uses a uid to identify a session
    uid = caffe.NCCL.new_uid()

    caffe.init_log()
    print 'Using devices %s' % str(gpus)

    procs = []
    for rank in range(len(gpus)):
        p = Process(target=solve,
                    args=(solver_path, snapshot, gpus, uid, rank))
        p.daemon = True
        p.start()
        procs.append(p)
    for p in procs:
        p.join()


def time(solver, nccl):
    fprop = []
    bprop = []
    total = caffe.Timer()
    allrd = caffe.Timer()
    for _ in range(len(solver.net.layers)):
        fprop.append(caffe.Timer())
        bprop.append(caffe.Timer())
    display = solver.param.display

    def show_time():
        if solver.iter % display == 0:
            s = '\n'
            for i in range(len(solver.net.layers)):
                s += 'forw %3d %8s ' % (i, solver.net._layer_names[i])
                s += ': %.2f\n' % fprop[i].ms
            for i in range(len(solver.net.layers) - 1, -1, -1):
                s += 'back %3d %8s ' % (i, solver.net._layer_names[i])
                s += ': %.2f\n' % bprop[i].ms
            s += 'solver total: %.2f\n' % total.ms
            s += 'allreduce: %.2f\n' % allrd.ms
            caffe.log(s)
            print s

    solver.net.before_forward(lambda layer: fprop[layer].start())
    solver.net.after_forward(lambda layer: fprop[layer].stop())
    solver.net.before_backward(lambda layer: bprop[layer].start())
    solver.net.after_backward(lambda layer: bprop[layer].stop())
    solver.add_callback(lambda: total.start(), lambda: (total.stop(), allrd.start()))
    solver.add_callback(nccl)
    solver.add_callback(lambda: '', lambda: (allrd.stop(), show_time()))

def plot(solver, nccl):

    display = solver.param.display
    maxIter = solver.param.max_iter


    # SET PLOTS DATA
    train_loss_C = zeros(maxIter/display)
    train_top1 = zeros(maxIter/display)
    train_top5 = zeros(maxIter/display)

    val_loss_C = zeros(maxIter/test_interval)
    val_top1 = zeros(maxIter/test_interval)
    val_top5 = zeros(maxIter/test_interval)


    it_axes = (arange(maxIter) * display) + display
    it_val_axes = (arange(maxIter) * test_interval) + test_interval

    _, ax1 = subplots()
    ax2 = ax1.twinx()
    ax1.set_xlabel('iteration')
    ax1.set_ylabel('train loss C (r), val loss C (y)')
    ax2.set_ylabel('train TOP1 (b), val TOP1 (g), train TOP-5 (c), val TOP-5 (k)')
    ax2.set_autoscaley_on(False)
    ax2.set_ylim([0, 1])

    lossC = np.zeros(maxIter)
    acc1 = np.zeros(maxIter)
    acc5 = np.zeros(maxIter)

    best_it = np.zeros(1)
    lowest_val_loss = np.zeros(1) + 1000

    def do_plot():

        if solver.iter % display == 0:
            lossC[solver.iter] = solver.net.blobs['loss3/loss3'].data.copy()
            acc1[solver.iter] = solver.net.blobs['loss3/top-1'].data.copy()
            acc5[solver.iter] = solver.net.blobs['loss3/top-5'].data.copy()

            loss_disp = 'loss3C= ' + str(lossC[solver.iter]) +  '  top-1= ' + str(acc1[solver.iter])

            print '%3d) %s' % (solver.iter, loss_disp)

            train_loss_C[solver.iter / display] = lossC[solver.iter]
            train_top1[solver.iter / display] = acc1[solver.iter]
            train_top5[solver.iter / display] = acc5[solver.iter]

            ax1.plot(it_axes[0:solver.iter / display], train_loss_C[0:solver.iter / display], 'r')
            ax2.plot(it_axes[0:solver.iter / display], train_top1[0:solver.iter / display], 'b')
            ax2.plot(it_axes[0:solver.iter / display], train_top5[0:solver.iter / display], 'c')

            ax1.set_ylim([0, 25])
            plt.title(training_id)
            plt.ion()
            plt.grid(True)
            plt.show()
            plt.pause(0.001)

            # VALIDATE Validation done this way only uses 1 GPU
        if solver.iter % test_interval == 0 and solver.iter > 0:
            loss_val_C = 0
            top1_val = 0
            top5_val = 0

            for i in range(test_iters):
                solver.test_nets[0].forward()
                loss_val_C += solver.test_nets[0].blobs['loss3/loss3'].data
                top1_val += solver.test_nets[0].blobs['loss3/top-1'].data
                top5_val += solver.test_nets[0].blobs['loss3/top-5'].data


            loss_val_C /= test_iters
            top1_val /= test_iters
            top5_val /= test_iters

            print("Val loss: " + str(loss_val_C))

            val_loss_C[solver.iter / test_interval - 1] = loss_val_C
            val_top1[solver.iter / test_interval - 1] = top1_val
            val_top5[solver.iter / test_interval - 1] = top5_val


            ax1.plot(it_val_axes[0:solver.iter / test_interval], val_loss_C[0:solver.iter / test_interval], 'y')
            ax2.plot(it_val_axes[0:solver.iter / test_interval], val_top1[0:solver.iter / test_interval], 'g')
            ax2.plot(it_val_axes[0:solver.iter / test_interval], val_top5[0:solver.iter / test_interval], 'k')


            ax1.set_ylim([0, 25])
            ax1.set_xlabel('iteration ' + 'Best it: ' + str(best_it[0]) + ' Best Val Loss: ' + str(int(lowest_val_loss[0])))
            plt.title(training_id)
            plt.ion()
            plt.grid(True)
            plt.show()
            plt.pause(0.001)
            title = '../../../ssd2/iMaterialistFashion/models/training/' + training_id + '_' + str(solver.iter) + '.png'
            savefig(title, bbox_inches='tight')

            if loss_val_C < lowest_val_loss[0]:
                print("Best Val loss!")
                lowest_val_loss[0] = loss_val_C
                best_it[0] = solver.iter
                filename = '../../../ssd2/iMaterialistFashion/models/CNN/' + training_id + 'best_valLoss_' + str(
                    int(lowest_val_loss[0])) + '_it_' + str(best_it[0]) + '.caffemodel'
                prefix = 30
                for cur_filename in glob.glob(filename[:-prefix] + '*'):
                    print(cur_filename)
                    os.remove(cur_filename)
                solver.net.save(filename)

    solver.add_callback(nccl)
    solver.add_callback(lambda: '', lambda: (do_plot()))


def solve(proto, snapshot, gpus, uid, rank):

    print 'Loading solver to GPU: ' + str(gpus[rank])

    caffe.set_mode_gpu()
    caffe.set_device(gpus[rank])
    caffe.set_solver_count(len(gpus))
    caffe.set_solver_rank(rank)
    caffe.set_multiprocess(True)

    solver = caffe.SGDSolver(proto)
    if snapshot and len(snapshot) != 0:
        print 'Loading snapshot from : ' + snapshot + '  to GPU: ' + str(gpus[rank])
        #solver.restore(snapshot)
        solver.net.copy_from(snapshot)
    else:
        print "Training from scratch"

    nccl = caffe.NCCL(solver, uid)
    nccl.bcast()
    if timing and rank == 0:
        print 'Timing ON'
        time(solver, nccl)
    else:
        solver.add_callback(nccl)

    if plotting and rank == 0:
        print 'Plotting ON'
        plot(solver, nccl)

    if solver.param.layer_wise_reduce:
        solver.net.after_backward(nccl)

    print 'Starting solver for GPU: ' + str(rank)
    solver.step(solver.param.max_iter)

train(solver_path, snapshot, gpus)