import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as compare_ssim

class Metrics:
    def __init__(self, imageA, imageB):
        self.imageA = np.array(imageA)
        self.imageB = np.array(imageB)
        self.imageA_gray = np.array(imageA.convert('L'))
        self.imageB_gray = np.array(imageB.convert('L'))

    def mse(self):
        # 计算MSE
        err = np.sum((self.imageA.astype("float") - self.imageB.astype("float")) ** 2)
        err /= float(self.imageA.shape[0] * self.imageA.shape[1])
        return err

    def psnr(self):
        # 计算MSE
        mse_value = self.mse()
        if mse_value == 0:
            return float('inf')  # MSE为0表示没有误差，PSNR为无穷大
        max_pixel = 255.0
        # 计算PSNR
        psnr_value = 20 * np.log10(max_pixel / np.sqrt(mse_value))
        return psnr_value

    def ssim(self):
        # 计算SSIM
        ssim_value, _ = compare_ssim(self.imageA_gray, self.imageB_gray, full=True)
        return ssim_value

    def ber(self):
        # 将图像数据转换为一维二进制数组
        binaryA = np.unpackbits(self.imageA.flatten().astype(np.uint8))
        binaryB = np.unpackbits(self.imageB.flatten().astype(np.uint8))

        # 计算二进制数据的误码
        total_bits = binaryA.size
        error_bits = np.sum(binaryA != binaryB)

        # 计算误码率
        ber_value = error_bits / total_bits
        return ber_value

    def calculate(self):
        # 计算所有的指标
        mse_value = self.mse()
        psnr_value = self.psnr()
        ssim_value = self.ssim()
        ber_value = self.ber()
        return {'MSE': mse_value, 'PSNR': psnr_value, 'SSIM': ssim_value, 'BER': ber_value}

# 使用示例
if __name__ == "__main__":
    # 从文件加载两个图像
    image1 = Image.open("path_to_image1.jpg")
    image2 = Image.open("path_to_image2.jpg")

    # 创建Metrics实例
    metrics = Metrics(image1, image2)
    # 计算并打印指标
    results = metrics.calculate()
    print(results)