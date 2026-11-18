# Xbox Series X Developer Mode — experimental bring-up

This is a **new Xbox UWP port target**, not an Xbox build of Mario Kart World yet.

## Stage 1: validate the graphics environment
1. On a Windows PC with Visual Studio, install **Universal Windows Platform development**, the Windows SDK, and C++ UWP tools.
2. Create a **C++/WinRT Blank App (Universal Windows)** project targeting x64. Name it `PlumberRacerXbox`.
3. Use the Direct3D test code in `D3D11Probe.cpp` as a starting point for device creation from your app's startup path.
4. Build an x64 UWP app package, deploy it to Xbox Dev Mode with the Device Portal, and inspect diagnostics.
5. Do **not** include copyrighted game files in the project or public builds.

## Why D3D11 first?
A D3D11 device smoke test establishes that the UWP app can access Xbox graphics. This does **not** imply the Switch 2 NVN2 renderer can be translated yet. The game port needs a real NVN2-to-Direct3D backend, ARM64 guest execution on x64, texture handling, input, memory management, and compatibility work.

## Important constraints
Xbox Developer Mode has UWP resource and API restrictions. The iOS Metal renderer and IPA cannot run directly on Xbox. Retail game installation is not supported by this target.

## Milestones
- [ ] x64 UWP app launches in Xbox Dev Mode
- [ ] D3D11 device and feature level reported
- [ ] Clear and present a swap chain to the TV
- [ ] Xbox controller input
- [ ] NVN2 graphics command translation
- [ ] Guest ARM64 execution and game boot

The probe is deliberately limited to device creation. It is **not** a playable port or a complete packaged app.
