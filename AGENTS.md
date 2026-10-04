# OpenCode Agent Instructions for PASS-Optimization

## Project Overview

This project implements the simulation for the paper:
**"PASS-Enabled Age-of-Information-Aware Wireless Status-Update Scheduling"**

## Key Components

- **PASS (Pinching-Antenna System)**: A dielectric waveguide with M candidate radiation points
- **K Sensing Devices**: Each with one-packet buffer and Bernoulli packet arrival
- **Age-of-Information (AoI)**: Metric to measure information freshness at the base station
- **Joint Scheduling**: Select device k and PASS point m to minimize weighted AoI

## System Model Summary

### State Definition (CRITICAL)

**State s(t) is defined AFTER packet generation at the beginning of slot t:**
- **∆k(t)**: BS AoI (from previous successful delivery) - does NOT reset on new arrival
- **bk(t)**: post-arrival buffer state (after Ak(t) generation)
- **δk(t)**: buffered packet age (after generation, δk=0 for new arrivals)
- **m-(t)**: inherited PASS configuration

### Time-Slot Events (per step()):

1. **Packet Generation**: Ak(t) ~ Bernoulli(λ) for each device
2. **State Observation**: Buffer states bk(t), BS AoI ∆k(t), inherited PASS config m-(t)
3. **Action Selection**: Choose (k, m) from feasible set A(t)
4. **PASS Reconfiguration**: If m-(t) ≠ m, incur switching overhead τsw = ρsw * τ
5. **Transmission**: Calculate SNR, determine success probability psuc
6. **State Updates**:
   - BS AoI: ∆k(t+1) = ∆k(t) + 1 - Yk(t)[∆k(t) - δk(t)]
   - Buffer: bk(t+1) = Ak(t+1) + [1 - Ak(t+1)]bk(t)[1 - Yk(t)]
   - Packet age: δk(t+1) = [1 - Ak(t+1)]bk(t)[1 - Yk(t)][δk(t) + 1]
   - PASS config: m-(t+1) = m if scheduled, else m-(t)
7. **Cost/Reward**: reward = -(weighted_AoI + η * violations)

### Key Equations:

- Power gain: Gk,m = β0 * (r0/rk,m)^α * exp(-αwg * ℓm)
- SNR threshold: γth = 2^(D/(B*τtx)) - 1
- Success probability: psuc = exp(-γth * σ² / (P * Gk,m))

## Configuration

- System parameters: `config/system_config.yaml`
- Agent parameters: `config/agent_config.yaml`

## Important Notes

- Use `reward = -cost` (negative cost as reward)
- State includes: BS AoI, device buffers, buffered packet ages, inherited PASS config
- Action space: K * M (device index * PASS point index)
- The environment follows Gym-style interface with reset() and step()
- **IMPORTANT**: BS AoI (∆k) and buffered packet age (δk) are SEPARATE quantities
  - BS AoI tracks freshness of information at base station
  - Buffered packet age tracks age of packet in device buffer

## File Structure

- `src/env/env.py`: Main environment with AntennaEnv class
- `src/env/env_classes.py`: Waveguide, Device, Packet classes
- `src/utils/geometry.py`: Distance and power gain calculations
- `src/utils/get_config.py`: Configuration loading