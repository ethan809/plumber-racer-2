#pragma once
// Xbox UWP gamepad state adapter. Wire to guest controller state in a later stage.
#include <windows.gaming.input.h>
namespace XboxNVN {
struct PadState {
    bool accelerate = false, brake = false, drift = false, item = false;
    float steer = 0.0f;
};
inline PadState ReadFirstPad() {
    PadState p{};
    auto pads = Windows::Gaming::Input::Gamepad::Gamepads;
    if (pads->Size == 0) return p;
    auto reading = pads->GetAt(0)->GetCurrentReading();
    using Windows::Gaming::Input::GamepadButtons;
    p.accelerate = reading.RightTrigger > 0.25;
    p.brake = reading.LeftTrigger > 0.25;
    p.steer = static_cast<float>(reading.LeftThumbstickX);
    p.drift = (reading.Buttons & GamepadButtons::RightShoulder) != GamepadButtons::None;
    p.item = (reading.Buttons & GamepadButtons::LeftShoulder) != GamepadButtons::None;
    return p;
}
} // namespace XboxNVN
