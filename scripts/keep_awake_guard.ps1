param(
    [int]$IntervalSeconds = 60
)

# Keeps Windows from going to sleep while the EEG run is active.
# The request is released automatically when this process exits.
Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;

public static class EegSleepGuard
{
    [DllImport("kernel32.dll", CharSet = CharSet.Auto, SetLastError = true)]
    public static extern uint SetThreadExecutionState(uint esFlags);

    public const uint ES_CONTINUOUS = 0x80000000;
    public const uint ES_SYSTEM_REQUIRED = 0x00000001;
}
"@

$flags = [EegSleepGuard]::ES_CONTINUOUS -bor [EegSleepGuard]::ES_SYSTEM_REQUIRED
[EegSleepGuard]::SetThreadExecutionState($flags) | Out-Null
try {
    while ($true) {
        Start-Sleep -Seconds $IntervalSeconds
    }
}
finally {
    [EegSleepGuard]::SetThreadExecutionState([EegSleepGuard]::ES_CONTINUOUS) | Out-Null
}
