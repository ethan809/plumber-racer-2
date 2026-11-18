// Xbox UWP startup: create a hardware D3D11 device, clear to blue and present.
// Graphics smoke test only; no game assets or Nintendo code.
#include <windows.applicationmodel.core.h>
#include <windows.ui.core.h>
#include <windows.graphics.display.h>
#include <windows.h>
#include <d3d11_1.h>
#include <dxgi1_2.h>
#include <wrl/client.h>
#include <cstdio>
#include <cmath>
#include "D3D11Probe.cpp"
#include "RomfsImageXbox.h"
#include "RecompGameDump.h"
#include <windows.storage.h>

using namespace Windows::ApplicationModel::Core;
using namespace Windows::UI::Core;
using Microsoft::WRL::ComPtr;

ref class XboxView sealed : IFrameworkView {
    CoreWindow^ window_;
public:
    virtual void Initialize(CoreApplicationView^) {}
    virtual void SetWindow(CoreWindow^ window) { window_ = window; }
    virtual void Load(Platform::String^) {}
    virtual void Uninitialize() {}
    virtual void Run() {
        window_->Activate();
        // Load the same DMROMFS1 image accepted by the iOS IPA.
        // Place romfs.img in this app's LocalFolder through Xbox Device Portal.
        XboxRomfs::Image gameImage;
        auto folder = Windows::Storage::ApplicationData::Current->LocalFolder->Path;
        std::wstring imagePath(folder->Data());
        imagePath += L"\\romfs.img";
        XboxRecomp::log(XboxRecomp::discover(std::wstring(folder->Data())));
        if (gameImage.open(imagePath)) {
            OutputDebugStringA("Xbox game launcher: DMROMFS1 mounted.\\n");
            if (gameImage.find("System/Resource"))
                OutputDebugStringA("Xbox game launcher: Mario RomFS marker found.\\n");
            else
                OutputDebugStringA("Xbox game launcher: Mario RomFS marker missing.\\n");
        } else {
            OutputDebugStringA("Xbox game launcher: ");
            OutputDebugStringA(gameImage.error.c_str());
            OutputDebugStringA("\\n");
        }
        auto gfx = XboxProbe::CreateDevice();
        ComPtr<IDXGIDevice1> dxgiDevice;
        if (FAILED(gfx.device.As(&dxgiDevice))) return;
        ComPtr<IDXGIAdapter> adapter;
        if (FAILED(dxgiDevice->GetAdapter(&adapter))) return;
        ComPtr<IDXGIFactory2> factory;
        if (FAILED(adapter->GetParent(IID_PPV_ARGS(&factory)))) return;
        DXGI_SWAP_CHAIN_DESC1 desc = {};
        desc.Width = 1280; desc.Height = 720;
        desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        desc.SampleDesc.Count = 1;
        desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        desc.BufferCount = 2;
        desc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL;
        ComPtr<IDXGISwapChain1> chain;
        if (FAILED(factory->CreateSwapChainForCoreWindow(gfx.device.Get(),
                reinterpret_cast<IUnknown*>(window_), &desc, nullptr, &chain))) return;
        ComPtr<ID3D11Texture2D> back;
        if (FAILED(chain->GetBuffer(0, IID_PPV_ARGS(&back)))) return;
        ComPtr<ID3D11RenderTargetView> rtv;
        if (FAILED(gfx.device->CreateRenderTargetView(back.Get(), nullptr, &rtv))) return;
        // Present continuously and expose actual swap-chain frame progress.
        // These are diagnostic frames, NOT Mario Kart World gameplay frames.
        LARGE_INTEGER frequency = {}, last = {}, now = {};
        QueryPerformanceFrequency(&frequency);
        QueryPerformanceCounter(&last);
        unsigned frames = 0;
        double elapsed = 0.0;
        while (true) {
            window_->Dispatcher->ProcessEvents(CoreProcessEventsOption::ProcessAllIfPresent);
            gfx.context->OMSetRenderTargets(1, rtv.GetAddressOf(), nullptr);
            // A slowly changing background makes successful frame presentation
            // visible even without a game graphics backend.
            float phase = static_cast<float>((frames % 240) / 240.0);
            const float diagnostic[4] = {
                0.04f + phase * 0.12f, 0.13f, 0.38f + phase * 0.25f, 1.0f
            };
            gfx.context->ClearRenderTargetView(rtv.Get(), diagnostic);
            HRESULT hr = chain->Present(1, 0);
            if (FAILED(hr)) {
                OutputDebugStringA("Xbox D3D11 Present failed.\\n");
                break;
            }
            ++frames;
            QueryPerformanceCounter(&now);
            if (frequency.QuadPart > 0) {
                double delta = double(now.QuadPart - last.QuadPart) /
                               double(frequency.QuadPart);
                elapsed += delta;
                last = now;
                if (elapsed >= 1.0) {
                    char message[128] = {};
                    std::snprintf(message, sizeof(message),
                        "Xbox D3D11 diagnostic: %u frames, %.1f FPS (not game frames)\\n",
                        frames, double(frames) / elapsed);
                    OutputDebugStringA(message);
                    frames = 0;
                    elapsed = 0.0;
                }
            }
        }
    }
};
ref class XboxFactory sealed : IFrameworkViewSource {
public:
    virtual IFrameworkView^ CreateView() { return ref new XboxView(); }
};
int main(Platform::Array<Platform::String^>^) {
    CoreApplication::Run(ref new XboxFactory());
    return 0;
}
