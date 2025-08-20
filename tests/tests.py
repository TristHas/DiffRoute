from imports import *

from diffroute.agg.kernel_aggregator import RoutingIRFAggregator
from diffroute.agg.kernel_sampler import SubResolutionSampler
from diffroute import LTIRouter

from .test_helpers import *
from .gen_data import generate_mini_data as generate_data
from .sequential_models import MODELS

###
### test SubResolutionSampler on random data
###
def test_subresolution_conv_1_channel(sample_mode="avg", n_time_steps=80, time_window=80, M=2):
    dt = 1/M
    sampler = SubResolutionSampler(dt, sample_mode)
    
    x_d = torch.rand(1,1,n_time_steps)
    x_h = sampler.phi(x_d)
    
    w_h = torch.randn(1,time_window*M) 
    #w_h = repeatedly_convolve(w_h, dt)
    w_d = sampler.phi_k(w_h)
    
    o_d = sampler.conv(x_d, w_d[None])
    o_h = sampler.conv(x_h, w_h[None])
    o_h = sampler.phi_inv(o_h)
    
    test = torch.allclose(o_h, o_d, rtol=10**-3, atol=10**-2)
    return test, o_h, o_d

def test_subresolution_conv_multi_channel(sample_mode="avg", C=32, n_time_steps=80, time_window=80, M=2):
    dt = 1/M
    sampler = SubResolutionSampler(dt, sample_mode)
    
    x_d = torch.randn(1,C,n_time_steps)
    x_h = sampler.phi(x_d)
    
    w_h = torch.randn(C,C,time_window*M) 
    #w_h = repeatedly_convolve(w_h, dt)
    w_d = sampler.phi_k(w_h.view(C*C,-1)).view(C,C,-1)

    o_d = sampler.conv(x_d, w_d)
    o_h = sampler.conv(x_h, w_h)
    o_h = sampler.phi_inv(o_h)
    
    test = torch.allclose(o_h, o_d, rtol=10**-3, atol=10**-2)
    
    return test, o_h, o_d

###
### test equivalence between model and conv version
###
def test_nosampling_conv_model(sample_mode="avg", model_name="pure_lag", 
                               num_timesteps=80, M=2,
                               time_window=80, device="cuda:7"):
    """
        Here include the test that still passes.
    """
    dt = 1/M
    sampler = SubResolutionSampler(dt, sample_mode)
    g, runoff_inputs = generate_data(num_timesteps)  # shape => (3, T)
    x = torch.tensor(runoff_inputs, dtype=torch.float32, device=device)  # shape (1, 3, T)
    x_fine = sampler.phi(x).cpu().numpy() # Zero holder hold upsampling
    
    out_seg = MODELS[model_name](x_fine, g, x_fine.shape[-1], dt=dt)  # shape (3, factor*T)

    
    out = run_dense_conv(x_fine, g, model=model_name, time_window=x_fine.shape[-1], dt=dt)

    test = torch.allclose(torch.from_numpy(out)[...,10:].float(), 
                          torch.from_numpy(out_seg)[...,10:].float(), 
                          rtol=10**-3, atol=10**-2)

    return test, out_seg, out

###
### test equivalence between model and conv version at sub-resolution
###
def test_subresolution_conv_model(sample_mode="avg", model_name="pure_lag", 
                                  num_timesteps=80, M=2,
                                  time_window=80, device="cuda:7"):
    """
    """
    dt = 1/M
    sampler = SubResolutionSampler(dt, sample_mode)
    g, runoff_inputs = generate_data(num_timesteps)  # shape => (3, T)
    x = torch.tensor(runoff_inputs, dtype=torch.float32, device=device)  # shape (1, 3, T)
    x_fine = sampler.phi(x).cpu().numpy() 

    
    out_seg = MODELS[model_name](x_fine, g, x_fine.shape[-1], dt=dt)  # shape (3, factor*T)
    out_seg = sampler.phi_inv(torch.tensor(out_seg)).numpy()

    out = run_dense_conv_subresolution(runoff_inputs, g, model_name, time_window=time_window, dt=dt)
    

    test = torch.allclose(torch.from_numpy(out)[...,10:].float(), 
                      torch.from_numpy(out_seg)[...,10:].float(), 
                      rtol=10**-3, atol=10**-2)
    return test, out, out_seg

###
### test equivalence between optimized kernel aggregation and manual.
###
def test_subresolution_irfagg(sample_mode="avg", model_name="pure_lag", 
                              time_window=80, 
                              M=2, device="cuda:7"):
    """"""
    g, _ = generate_data(time_window)

    dt = 1/M
    kernel_ref = gen_dense_k_subresolution(g, time_window=time_window, 
                                           model=model_name, dt=dt)
    kernel_ref = torch.from_numpy(kernel_ref).to(device)
    params = read_params(g, model_name).to(device)
    agg = RoutingIRFAggregator(g, max_delay=time_window, 
                               block_size=4,
                               irf_fn=model_name, 
                               irf_agg="log_triton", 
                               index_precomp="optimized", 
                               include_index_diag=True,
                               dt=dt, sampling_mode=sample_mode).to(device)
    kernel = agg(params).to_dense()
    test = torch.allclose(kernel_ref, kernel,
                          rtol=10**-3, atol=10**-2)
    return test, kernel_ref, kernel

###
### Test dense convolution with optimized kernel aggregation
###
def test_subresolution_conv_with_agg(sample_mode="avg", model_name="pure_lag", 
                                     num_timesteps=80, M=2,
                                     time_window=80, device="cuda:7"):
    """
    """
    dt = 1/M
    sampler = SubResolutionSampler(dt, sample_mode)
    g, runoff_inputs = generate_data(num_timesteps)  # shape => (3, T)
    x = torch.tensor(runoff_inputs, dtype=torch.float32, device=device)  # shape (1, 3, T)
    
    x_fine = sampler.phi(x).cpu().numpy() 
    out_seg = MODELS[model_name](x_fine, g, x_fine.shape[-1], dt=dt)  # shape (3, factor*T)
    out_seg = sampler.phi_inv(torch.tensor(out_seg)).numpy()

    params = read_params(g, model_name).to(device)
    agg = RoutingIRFAggregator(g, max_delay=time_window, 
                               block_size=16,
                               irf_fn=model_name, 
                               irf_agg="log_triton", 
                               index_precomp="optimized", 
                               include_index_diag=True,
                               dt=dt, sampling_mode=sample_mode).to(device)
    w = agg(params).to_dense()
    out = conv(x, w).cpu().numpy()

    test = torch.allclose(torch.from_numpy(out)[...,10:].float(), 
                          torch.from_numpy(out_seg)[...,10:].float(), 
                          rtol=10**-3, atol=10**-2)
    return test, out, out_seg

###
### Test full block sparse pipeline
###
def test_subresolution_full_module(sample_mode="avg", model_name="pure_lag", 
                                   num_timesteps=80, M=2,
                                   time_window=80, device="cuda:7"):
    """
    """
    dt = 1/M
    sampler = SubResolutionSampler(dt, sample_mode)
    g, runoff_inputs = generate_data(num_timesteps)  # shape => (3, T)
    x = torch.tensor(runoff_inputs, dtype=torch.float32, device=device)  # shape (1, 3, T)
    x_fine = sampler.phi(x).cpu().numpy() 
    out_seg = MODELS[model_name](x_fine, g, x_fine.shape[-1], dt=dt)  # shape (3, factor*T)
    out_seg = sampler.phi_inv(torch.tensor(out_seg)).numpy()

    params = read_params(g, model_name).to(device)
    model = LTIRouter(g, 
                     max_delay=time_window, 
                     block_size=16,
                     irf_fn=model_name, 
                     irf_agg="log_triton", 
                     index_precomp="optimized", 
                     runoff_to_output=False,
                     dt=dt, sampling_mode=sample_mode).to(device)
    out = model(x[None], params).cpu().numpy()

    test = torch.allclose(torch.from_numpy(out)[...,10:].float(), 
                          torch.from_numpy(out_seg)[...,10:].float(), 
                          rtol=10**-3, atol=10**-2)
    return test, out, out_seg

def run_test_suits(sample_mode="avg", model_name="pure_lag", M=1, **kwargs):
    test = test_subresolution_conv_1_channel(sample_mode=sample_mode, M=M)[0]
    print(f"test_subresolution_conv_1_channel: {test}")
    test = test_subresolution_conv_multi_channel(sample_mode=sample_mode, M=M)[0]
    print(f"test_subresolution_conv_multi_channel: {test}")
    test = test_nosampling_conv_model(sample_mode=sample_mode, model_name=model_name, M=M)[0]
    print(f"test_nosampling_conv_model: {test}")
    test = test_subresolution_conv_model(sample_mode=sample_mode, model_name=model_name, M=M)[0]
    print(f"test_subresolution_conv_model: {test}")
    test = test_subresolution_irfagg(sample_mode=sample_mode, model_name=model_name, M=M)[0]
    print(f"test_subresolution_irfagg: {test}")
    test = test_subresolution_conv_with_agg(sample_mode=sample_mode, model_name=model_name, M=M)[0]
    print(f"test_subresolution_conv_with_agg: {test}")
    test = test_subresolution_full_module(sample_mode=sample_mode, model_name=model_name, M=M)[0]
    print(f"test_subresolution_full_module: {test}")

if __name__ == "__main__":
    for model in ["muskingum", "hayami", "pure_lag", "linear_storage", "nash_cascade"]:
        for M in [1, 10]:
            print(model, M)
            run_test_suits(model_name=model, M=M)