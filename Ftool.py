import torch

# spectrum processing module
def half_chaifen(input, size):
    fft2tensor = torch.fft.fft2(input)
    fft2tensor = torch.fft.fftshift(fft2tensor)
    real_a = torch.real(fft2tensor)
    imag_a = torch.imag(fft2tensor)
    preserve_row_real = real_a[:, :, 0:1, :]
    preserve_column_real = real_a[:, :, 1:size, 0:1]
    preserve_row_imag = imag_a[:, :, 0:1, :]
    preserve_column_imag = imag_a[:, :, 1:size, 0:1]
    A_real = real_a[:, :, :, 1:int(size / 2 + 1)]
    A_imag = imag_a[:, :, :, 1:int(size / 2 + 1)]
    return A_real, A_imag, preserve_row_real, preserve_column_real, preserve_row_imag, preserve_column_imag

# spectrum inverse processing module
def half_hecheng(input_real, input_imag, preserve_row_real, preserve_column_real, preserve_row_imag,
                 preserve_column_imag, size):
    A_real = input_real[:, :, 1:size, 0:int(size / 2)]
    A_imag = input_imag[:, :, 1:size, 0:int(size / 2)]
    B_real = torch.rot90(A_real, dims=[2, 3])
    B_real = torch.rot90(B_real, dims=[2, 3])
    B_real = B_real[:, :, :, 1:int(size / 2)]
    B_imag = -torch.rot90(A_imag, dims=[2, 3])
    B_imag = torch.rot90(B_imag, dims=[2, 3])
    B_imag = B_imag[:, :, :, 1:int(size / 2)]
    real = torch.cat([preserve_row_real, torch.cat([preserve_column_real, A_real, B_real], dim=3)], dim=2)
    imag = torch.cat([preserve_row_imag, torch.cat([preserve_column_imag, A_imag, B_imag], dim=3)], dim=2)
    complex = torch.complex(real, imag)
    complex = torch.fft.ifftshift(complex)
    complex = torch.fft.ifft2(complex)
    return complex
