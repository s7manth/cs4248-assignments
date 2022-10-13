import torch


def question1(shape):
    """
    Returns a tensor with all elements are number one with a given shape
    """
    return torch.ones(shape)


def question2(data):
    """
    Converts a python list into a pytorch tensor with @code{torch.long} data type
    """
    return torch.tensor(data).to(torch.long)


def question3(a, b):
    """
    Computes 3 * a + 2 * b
    """
    return 3 * a + 2 * b


def question4(a):
    """
    Retrieves the last column from a 2 dimensional tensor
    """
    return a[..., -1]


def question5(data):
    """
    Combines a list of tensors into a new tensor at the last dimension, then expands the 1st dimension
    """
    st = torch.stack(data, -1)
    result = st.expand([1] + list(st.size()))

    return result


def question6(data):
    """
    Combines a list of 1-D tensors with different lengths into a new tensor by padding the shorter
    tensors with 0 on the left side
    """
    max_length = max(map(len, data))
    data = list(map(lambda x: torch.cat((torch.zeros(max_length - x.size(dim=0), dtype=torch.long), x), dim=0), data))
    return torch.stack(data)


def question7(y, w, b):
    """
    Define a function that calculates w * (y - b)
    """
    return torch.matmul(torch.transpose(w, 0, 1), y - b)


def question8(y, w, b):
    """
    Calculates batch w * (y - b)
    """
    return torch.bmm(torch.transpose(w, 1, 2), y - b)


def question9(x):
    """
    Calculates the mean along the n dimension without accounting for the 0-values
    """
    mask = x != 0
    return (x * mask).sum(dim=2) / mask.sum(dim=2)


def question10(pairs):
    """
    Calculates the Euclidean distance of each vector pair
    """
    return torch.diagonal(torch.cdist(torch.tensor([x[0] for x in pairs]), torch.tensor([x[1] for x in pairs]),
                                      p=2.0), 0)


def main():
    q1_input = (2, 3)
    print('Q1 Example input: {}\n'.format(q1_input))

    q1 = question1(q1_input)
    print('Q1 example output: \n{}\n'.format(q1))

    q2_input = [[1., 2.1, 3.0], [4., 5., 6.2]]
    print('Q2 Example input: \n{}\n'.format(q2_input))

    q2 = question2(q2_input)
    print('Q2 example output: \n{}\n'.format(q2))
    print('Q3 Example input: \na: {}\nb: {}\n'.format(q2, question2([[1, 1, 1], [1, 1, 1]])))

    q3 = question3(q2, question2([[1,1,1], [1,1,1]]))
    print('Q3 example output: \n{}\n'.format(q3))
    print('Q4 Example input: \n{}\n'.format(q2))
    print('Q4 example output: \n{}\n'.format(question4(q2)))

    q5_input = [question4(q1).type(torch.long), question4(q2), question4(q3)]
    print('Q5 Example input: \n{}\n'.format(q5_input))

    q5 = question5(q5_input)
    print('Q5 example output: \n{}\n'.format(q5))

    q6_input = [question2([1]), question2([2, 2]), question2([3, 3, 3])]
    print('Q6 Example input: \n{}\n'.format(q6_input))

    q6 = question6(q6_input)
    print('Q6 example output: \n{}\n'.format(q6))

    q7_input = (torch.tensor([[1.12, 1.57, 2.11], [0.23, 0.72, 1.19], [0.52, 0.4, 0.31]]),
                torch.tensor([[0.3, 0.2], [0.6, -0.1], [-0.3, 0.2]]),
                torch.tensor([[0.02, -0.03, 0.01], [0.03, 0.02, -0.01], [0.02, 0, 0.01]]))
    print('Q7 Example input \ny: \n{}\nw: {}\nb: {}\n'.format(*q7_input))

    q7 = question7(*q7_input)
    print('Q7 example output: \n{}\n'.format(q7))

    q8_input = (torch.tensor([[[1.12, 1.57, 2.11], [0.23, 0.72, 1.19], [0.52, 0.4, 0.31]], [[0.31, 0.36, 0.3], [0.82, 0.51, 0.78], [0.21, -0.01, 0.22]]]),
                torch.tensor([[[0.3, 0.2], [0.6, -0.1], [-0.3, 0.2]], [[-0.6, 0.5], [0.1, 0.2], [0.4, -0.2]]]),
                torch.tensor([[[0.02, -0.03, 0.01], [0.03, 0.02, -0.01], [0.02, 0, 0.01]],[[0.01, -0.04, 0.0], [0.02, 0.01, -0.02], [0.01, -0.01, 0.02]]]))
    print('Q8 Example input \ny: \n{}\nw: {}\nb: {}\n'.format(*q8_input))

    q8 = question8(*q8_input)
    print('Q8 example output: \n{}\n'.format(q8))

    q9_input = torch.tensor([[[1.0, 0., 0.], [1.2, 0., 0.]], [[2.0, 2.2, 0.], [2.2, 2.6, 0.]], [[3.0, 3.2, 3.1], [3.2, 3.4, 3.6]]])
    print('Q9 Example input: \n{}\n'.format(q9_input))

    q9 = question9(q9_input)
    print('Q9 example output: \n{}\n'.format(q9))

    q10_input = [([1, 1, 1], [2, 2, 2]), ([1, 2, 3], [3, 2, 1]), ([0.1, 0.2, 0.3], [0.33, 0.25, 0.1])]
    print('Q10 Example input: \n{}\n'.format(q10_input))

    q10 = question10(q10_input)
    print('Q10 example output: \n{}\n'.format(q10))

    print('\n==== A2 Part 1 Done ====')


if __name__ == "__main__":
    main()