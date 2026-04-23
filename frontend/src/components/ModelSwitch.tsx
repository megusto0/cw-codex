import { useModelPreference } from '../model-context';

export default function ModelSwitch() {
  const { model, options, setModel } = useModelPreference();
  const active = options.find((item) => item.key === model) ?? options[0];

  return (
    <div className="model-switch">
      <div className="switch-header">
        <span className="field-label">Модель</span>
        <span className="switch-caption">{active.scientificDescription}</span>
      </div>
      <div className="segmented-control" role="tablist" aria-label="Выбор модели retrieval">
        {options.map((option) => (
          <button
            key={option.key}
            type="button"
            role="tab"
            aria-selected={model === option.key}
            className={`segment-button${model === option.key ? ' active' : ''}`}
            onClick={() => setModel(option.key)}
          >
            {option.label}
          </button>
        ))}
      </div>
    </div>
  );
}
