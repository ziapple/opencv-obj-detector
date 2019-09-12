import matplotlib.pyplot as plt
import numpy


class GD(object):

    def __init__(self, seed=None, precision=1.E-6):
        self.seed = GD.get_seed(seed)  # 梯度下降算法的种子点
        self.prec = precision  # 梯度下降算法的计算精度

        self.path = list()  # 记录种子点的路径及相应的目标函数值
        self.solve()  # 求解主体
        self.display()  # 数据可视化展示

    def solve(self):
        x_curr = self.seed
        val_curr = GD.func(*x_curr)
        self.path.append((x_curr, val_curr))

        omega = 1
        while omega > self.prec:
            x_delta = omega * GD.get_grad(*x_curr)
            x_next = x_curr - x_delta  # 沿梯度反向迭代
            val_next = GD.func(*x_next)

            if numpy.abs(val_next - val_curr) < self.prec:
                break

            if val_next < val_curr:
                x_curr = x_next
                val_curr = val_next
                omega *= 1.2
                self.path.append((x_curr, val_curr))
            else:
                omega *= 0.5

    def display(self):
        print('Iteration steps: {}'.format(len(self.path)))
        print('Seed: ({})'.format(', '.join(str(item) for item in self.path[0][0])))
        print('Solution: ({})'.format(', '.join(str(item) for item in self.path[-1][0])))

        fig = plt.figure(figsize=(10, 4))

        ax1 = plt.subplot(1, 2, 1)
        ax2 = plt.subplot(1, 2, 2)

        ax1.plot(numpy.array(range(len(self.path))) + 1, numpy.array(list(item[1] for item in self.path)), 'k.')
        ax1.plot(1, self.path[0][1], 'go', label='starting point')
        ax1.plot(len(self.path), self.path[-1][1], 'r*', label='solution')
        ax1.set(xlabel='$iterCnt$', ylabel='$iterVal$')
        ax1.legend()

        x = numpy.linspace(-100, 100, 500)
        y = numpy.linspace(-100, 100, 500)
        x, y = numpy.meshgrid(x, y)
        z = GD.func(x, y)
        ax2.contour(x, y, z, levels=36)

        x2 = numpy.array(list(item[0][0] for item in self.path))
        y2 = numpy.array(list(item[0][1] for item in self.path))
        ax2.plot(x2, y2, 'k--', linewidth=2)
        ax2.plot(x2[0], y2[0], 'go', label='starting point')
        ax2.plot(x2[-1], y2[-1], 'r*', label='solution')

        ax2.set(xlabel='$x$', ylabel='$y$')
        ax2.legend()

        fig.tight_layout()
        fig.savefig('test_plot.png', dpi=500)

        plt.show()
        plt.close()

    # 内部种子生成函数
    @staticmethod
    def get_seed(seed):
        if seed is not None:
            return numpy.array(seed)
        return numpy.random.uniform(-100, 100, 2)

    # 目标函数
    @staticmethod
    def func(x, y):
        return 5 * x ** 2 + 2 * y ** 2 + 3 * x - 10 * y + 4

    # 目标函数的归一化梯度
    @staticmethod
    def get_grad(x, y):
        grad_ori = numpy.array([10 * x + 3, 4 * y - 10])
        length = numpy.linalg.norm(grad_ori)
        if length == 0:
            return numpy.zeros(2)
        return grad_ori / length


if __name__ == '__main__':
    GD()