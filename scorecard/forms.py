from django import forms
from .models import ClientType, BusinessType, Scorecard

class ScorecardForm(forms.ModelForm):
    class Meta:
        model = Scorecard
        fields = ('client_type', 'business_type', 'scorecard')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.fields['client_type'].queryset = ClientType.objects.none()
        self.fields['business_type'].queryset = BusinessType.objects.none()

        if 'client_type' in self.data and 'business_type' in self.data:
            try:
                client_type_id = int(self.data.get('client_type'))
                business_type_id = int(self.data.get('business_type'))

                self.fields['scorecard'].queryset = Scorecard.objects.filter(client_type_id=client_type_id, business_type_id=business_type_id).order_by('name')
            
            except (ValueError, TypeError):
                pass  # invalid input from the client; ignore and fallback to empty City queryset
        elif self.instance.pk:
            self.fields['scorecard'].queryset = self.instance.Scorecard.order_by('name')
