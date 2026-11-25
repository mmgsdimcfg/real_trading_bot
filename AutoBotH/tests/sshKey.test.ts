// [2026-11-25] key login: a path to the public key means the private key beside it
import { describe, expect, it } from 'vitest';
import { privateKeyPath } from '../src/main/ssh';

describe('privateKeyPath', () => {
  const have = (...files: string[]) => (p: string) => files.includes(p);
  it('uses the private key beside a .pub path', () => {
    expect(privateKeyPath('C:\\Users\\me\\.ssh\\id_ed25519.pub', have('C:\\Users\\me\\.ssh\\id_ed25519'))).toBe('C:\\Users\\me\\.ssh\\id_ed25519');
    expect(privateKeyPath('/home/me/.ssh/id_ed25519.PUB', have('/home/me/.ssh/id_ed25519'))).toBe('/home/me/.ssh/id_ed25519');
  });
  it('keeps the path when there is no private key beside it, or it is not a .pub path', () => {
    expect(privateKeyPath('C:\\k\\id.pub', have())).toBe('C:\\k\\id.pub');
    expect(privateKeyPath('C:\\k\\id_ed25519', have('C:\\k\\id_ed25519'))).toBe('C:\\k\\id_ed25519');
    expect(privateKeyPath('C:\\k\\key.pem', have())).toBe('C:\\k\\key.pem');
  });
});
