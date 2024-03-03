import numpy as np
class MaxPool2:
    def iterate_regions(self, image):
        h, w, _ = image.shape
        new_h = h // 2 # // 연산은 나눗셈의 몫을 가져옴, 즉 나눗셈 결과의 정수만 가져옴
        new_w = w // 2

        for i in range(new_h):
            for j in range(new_w):
                im_region = image[(i * 2):(i * 2 + 2), (j * 2):(j * 2 + 2)]
                yield im_region, i, j
    
    def forward(self, input):
        self.last_input = input # Back propagation에서 사용되므로, 지금 몰라도 됨

        h, w, num_filters = input.shape
        output = np.zeros((h // 2, w // 2, num_filters))

        for im_region, i, j in self.iterate_regions(input):
            output[i, j] = np.amax(im_region, axis=(0, 1)) # array에서 max값 찾기

        return output