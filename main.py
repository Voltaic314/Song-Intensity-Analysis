from song import Song


def main():
    # Example usage
    song1 = Song(filepath='example songs\\Maneli Jamal - Dire, Dire Docks (from Super Mario 64).mp3', title='Dire, Dire Docks', artist='Maneli Jamal') # lighter song
    song2 = Song(filepath='example songs\\David Guetta & Bebe Rexha - I\'m Good (Blue).mp3', title='I\'m Good (Blue)', artist='David Guetta & Bebe Rexha') # heavier song
    
    heavy_score1 = song1.calculate_heavy_score(num_sections=10)
    heavy_score2 = song2.calculate_heavy_score(num_sections=10)
    
    print(f'Heavy score for {song1}: {heavy_score1}')
    print(f'Heavy score for {song2}: {heavy_score2}')


if __name__ == '__main__':
    main()