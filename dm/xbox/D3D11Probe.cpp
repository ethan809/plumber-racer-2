// Xbox UWP D3D11 device-creation probe.
// Add to an x64 C++/WinRT UWP project and call XboxProbe::CreateDevice()
// from an appropriate startup callback. Does not render a frame.
#include <d3d11_1.h>
#include <dxgi1_2.h>
#include <wrl/client.h>
#include <stdexcept>
#include <string>
#include <cstdio>

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")

namespace XboxProbe {
using Microsoft::WRL::ComPtr;

struct DeviceInfo {
    ComPtr<ID3D11Device1> device;
    ComPtr<ID3D11DeviceContext1> context;
    D3D_FEATURE_LEVEL level{};
};

inline DeviceInfo CreateDevice() {
    constexpr D3D_FEATURE_LEVEL levels[] = {
        D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0,
        D3D_FEATURE_LEVEL_10_1, D3D_FEATURE_LEVEL_10_0
    };
    ComPtr<ID3D11Device> device;
    ComPtr<ID3D11DeviceContext> context;
    D3D_FEATURE_LEVEL selected{};
    const HRESULT hr = D3D11CreateDevice(
        nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr,
        D3D11_CREATE_DEVICE_BGRA_SUPPORT,
        levels, static_cast<UINT>(sizeof(levels) / sizeof(levels[0])),
        D3D11_SDK_VERSION, &device, &selected, &context);
    if (FAILED(hr)) {
        char msg[128];
        std::snprintf(msg, sizeof(msg), "D3D11CreateDevice failed: HRESULT 0x%08lX",
                      static_cast<unsigned long>(hr));
        throw std::runtime_error(msg);
    }
    DeviceInfo out;
    out.level = selected;
    if (FAILED(device.As(&out.device)) || FAILED(context.As(&out.context))) {
        throw std::runtime_error("D3D11.1 interfaces unavailable");
    }
    return out;
}
} // namespace XboxProbe
