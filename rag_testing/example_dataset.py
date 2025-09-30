examples = [
    {
        "inputs": {"question": "Do cats love milk?"},
        "reference_outputs": {"answer": "Cats can drink milk, but many are lactose intolerant and it can cause digestive issues."},
        "reference_retrieved_knowledge": ['''Foods that should not be given to cats include onions, garlic, 
        green tomatoes, raw potatoes, chocolate, grapes, and raisins. Though milk is not toxic, it can cause an upset stomach and gas. Tylenol and aspirin 
        are extremely toxic to cats, as are many common houseplants. Feeding cats dog food or canned tuna 
        that’s for human consumption can cause malnutrition.''',
        '''
            Many cats cannot properly digest cow’s milk. Milk and milk products give them diarrhea.
        ''']
    },
    {
        "inputs": {"question": "What diseases do cats catch?"},
        "reference_outputs": {"answer": "Cats can catch all sorts of diseases such as gum disease, canine heart worms, and even cancer and AIDS. They can also get tapeworms."},
        "reference_retrieved_knowledge": [
            '''Cats are subject to gum disease and to dental caries. They should have their teeth cleaned by the vet or the cat dentist once a year.''',
            '''Cats, especially older cats, do get cancer. Many times this disease can be treated successfully. ''',
            '''Cats can get tapeworms from eating fleas. These worms live inside the cat forever, or until they are 
            removed with medication. They reproduce by shedding a link from the end of their long bodies. This link crawls out 
            the cat’s anus, and sheds hundreds of eggs. These eggs are injected by flea larvae, and the cycle continues. Humans 
            may get these tapeworms too, but only if they eat infected fleas. Cats with tapeworms should be dewormed by a veterinarian.
            Cats can get tapeworms from eating mice. If your cat catches a mouse it is best to take the prize away from it.
            Though rare, cats can contract canine heart worms. ''']
    },
]