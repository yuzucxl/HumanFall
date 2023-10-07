import numpy as np

def decode_conv(y):
    # shape = (4,len(y)/2)
    # 初始化
    score_list = np.array([[float('inf') for i in range(int(len(y) / 2) + 1)] for i in range(4)])
    for i in range(4):
        score_list[i][0] = 0
    # 记录回溯路径
    trace_back_list = []
    # 每个阶段的回溯块
    trace_block = []
    # 4种状态 0-3分别对应['00','01','10','11']
    states = ['00', '01', '10', '11']

    # 根据不同 状态 和 输入 编码信息
    def encode_with_state(x, state):
        # 编码后的输出
        y = []
        u_1 = 0 if state <= 1 else 1
        u_2 = state % 2
        c_1 = (x + u_1 + u_2) % 2
        c_2 = (x + u_2) % 2
        y.append(c_1)
        y.append(c_2)
        return y

    # 计算汉明距离
    def hamming_dist(y1, y2):
        dist = (y1[0] - y2[0]) % 2 + (y1[1] - y2[1]) % 2
        return dist

    # 根据当前状态now_state和输入信息比特input，算出下一个状态
    def state_transfer(input, now_state):
        u_1 = int(states[now_state][0])
        next_state = f'{input}{u_1}'
        return states.index(next_state)

    # 根据不同初始时刻更新参数
    # 也即指定状态为 state 时的参数更新
    # y_block 为 y 的一部分， shape=(2,)
    # pre_state 表示当前要处理的状态
    # index 指定需要处理的时刻
    def update_with_state(y_block, pre_state, index):
        # 输入的是 0
        encode_0 = encode_with_state(0, pre_state)
        next_state_0 = state_transfer(0, pre_state)
        score_0 = hamming_dist(y_block, encode_0)
        # 输入为0，且需要更新
        if score_list[pre_state][index] + score_0 < score_list[next_state_0][index + 1]:
            score_list[next_state_0][index + 1] = score_list[pre_state][index] + score_0
            trace_block[next_state_0][0] = pre_state
            trace_block[next_state_0][1] = 0
        # 输入的是 1
        encode_1 = encode_with_state(1, pre_state)
        next_state_1 = state_transfer(1, pre_state)
        score_1 = hamming_dist(y_block, encode_1)
        # 输入为1，且需要更新
        if score_list[pre_state][index] + score_1 < score_list[next_state_1][index + 1]:
            score_list[next_state_1][index + 1] = score_list[pre_state][index] + score_1
            trace_block[next_state_1][0] = pre_state
            trace_block[next_state_1][1] = 1
        if pre_state == 3 or index == 0:
            trace_back_list.append(trace_block)

    # 默认寄存器初始为 00。也即，开始时刻，默认状态为00
    # 开始第一个 y_block 的更新
    y_block = y[0:2]
    trace_block = [[-1, -1] for i in range(4)]
    update_with_state(y_block=y_block, pre_state=0, index=0)
    # 开始之后的 y_block 更新
    for index in range(2, int(len(y)), 2):
        y_block = y[index:index + 2]
        for state in range(len(states)):
            if state == 0:
                trace_block = [[-1, -1] for i in range(4)]
            update_with_state(y_block=y_block, pre_state=state, index=int(index / 2))
    # 完成前向过程，开始回溯
    # state_trace_index 表示 开始回溯的状态是啥
    state_trace_index = np.argmin(score_list[:, -1])
    # 记录原编码信息
    x = []
    for trace in range(len(trace_back_list) - 1, -1, -1):
        x.append(trace_back_list[trace][state_trace_index][1])
        state_trace_index = trace_back_list[trace][state_trace_index][0]
    x = list(reversed(x))
    print(y, "解码为：", x)
    return x


# 测试代码
if __name__ == '__main__':
    # 对应 1 1 0 0 0
    decode_conv([0, 1, 1, 1, 0, 1, 1, 1, 0, 0])