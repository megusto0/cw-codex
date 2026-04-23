import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { describe, expect, it } from 'vitest';
import ModelSwitch from '../components/ModelSwitch';
import { ModelProvider, useModelPreference } from '../model-context';

function ModelValue() {
  const { model } = useModelPreference();
  return <div data-testid="selected-model">{model}</div>;
}

describe('model switch', () => {
  it('persists selection in localStorage and reacts to URL model param', async () => {
    window.localStorage.clear();

    render(
      <MemoryRouter initialEntries={['/?model=som']}>
        <ModelProvider>
          <ModelSwitch />
          <ModelValue />
        </ModelProvider>
      </MemoryRouter>,
    );

    expect(screen.getByTestId('selected-model')).toHaveTextContent('som');

    fireEvent.click(screen.getByRole('tab', { name: 'Сиамская temporal-модель' }));

    await waitFor(() => {
      expect(screen.getByTestId('selected-model')).toHaveTextContent('siamese_temporal');
      expect(window.localStorage.getItem('cw-codex:model')).toBe('siamese_temporal');
    });
  });
});
