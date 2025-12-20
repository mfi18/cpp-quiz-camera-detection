
#include <SFML/Graphics.hpp> // For Rendering and Basic UI
#include <SFML/Audio.hpp> // For Audio/sound effects
#include <SFML/Window/Event.hpp> // For Keys pressed, mouse detection, window closing
#include <iostream> // Input/Output for Cpp
#include <cstdint> // For fixed-width integers types (uin8t)
#include <cstdlib> // For rand(), srand()
#include <ctime> // For time
#include <vector> // For Dynamic arrays
#include <fstream> // For reading from files
#include <sstream> // For string stream processing
#include <optional> // For handling events safely
#include <algorithm> // Randomizing question/answer order
#include <random> // Used for shuffle()
#include <chrono> // For Clock and Time functions
#include <memory> // For saving memory, deletes background image to save memory leaks
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace sf;
using namespace cv;

// Game Constants
const int WINDOW_WIDTH = 1600;
const int WINDOW_HEIGHT = 900;
const float TIME_PER_QUESTION = 15.0f;

// Colors
const Color BACKGROUND_COLOR(30, 0, 60);
const Color UI_BASE_COLOR(25, 10, 50, 150);
const Color CORRECT_COLOR(40, 180, 99);
const Color INCORRECT_COLOR(231, 76, 60);
const Color DEFAULT_OUTLINE_COLOR(150, 100, 255);

// Game States
enum GameState {
    MENU,
    SELECT_DIFFICULTY,
    SET_LIMIT,
    SETTINGS,
    QUIZ_MODE,
    PAUSED,
    GAME_OVER
};

// Data Structure
struct QuizQuestion {
    string questionText;
    vector<string> options; // Dynamic list to store 4 options
    int correctAnswerIndex;
    int userSelectedOption = -1; // Remembers what user clicked (-1 means nothing)
};

// Particles for wrong answers
struct Particle {
    RectangleShape shape;
    Vector2f velocity; // Speed in x y directions of particles, vector2f holds two floating point numbers
    float lifetime; // For Countdown
};

// Floating Text (+1) for correct answer
struct FloatingText {
    Text text;
    float lifetime;
    float speed;
    FloatingText(const Font& font, const string& str, float x, float y) // Constructor
        : text(font, str, 30)
    {
        text.setPosition({ x, y });
        text.setFillColor(Color::Green);
        text.setOutlineColor(Color::White);
        text.setOutlineThickness(2);
        lifetime = 1.0f;
        speed = 100.0f;
    }
};

// --------------------------------------------------------
//            NEW CLASS: HAND GESTURE TRACKER
// --------------------------------------------------------


class GestureTracker {
public:
    VideoCapture cap;
    int detectedFingers = 0;
    // Logic for "Holding" a gesture
    int lastStableCount = 0;
    float holdTime = 0.0f;
    const float REQUIRED_HOLD_TIME = 0.2f; // Must hold gesture for 1 second to trigger
    bool triggerAction = false; // True when action should fire
    bool isWindowOpen = false;
    bool isActive = false;

    GestureTracker() {
        //Initially Closed
    }

    ~GestureTracker() {
        stopCamera();
    }

    void setEnabled(bool enabled) {
        if (enabled) {
            if (!cap.isOpened()) {
                cap.open(0); // Try default
                if (!cap.isOpened()) cap.open(1); // Try secondary
            }
        }
        else {
            stopCamera();
        }
    }

    void stopCamera() {
        if (cap.isOpened()) cap.release();
        if (isWindowOpen) {
            try { destroyWindow("Gesture Control"); }
            catch (...) {}
            isWindowOpen = false;
        }
    }

    void update(float dt, bool isActive) {
        if (!cap.isOpened()) return;
        if (!isActive) {
            if (isWindowOpen) {
                try {
                    destroyWindow("Gesture Control");
                }
                catch (...) {}
                isWindowOpen = false;
            }
            return;
        }
        Mat frame, hsv, mask, drawing;
        cap >> frame;
        if (frame.empty()) return;

        // Flip frame for mirror effect
        flip(frame, frame, 1);

        // Define Region of Interest (ROI) - User puts hand in a box
        // Using a fixed box ensures better lighting consistency
        cv::Rect roiRect(50, 50, 300, 300);
        rectangle(frame, roiRect, Scalar(255, 0, 0), 2);

        Mat roi = frame(roiRect);

        // 1. Convert to HSV for skin detection
        cvtColor(roi, hsv, COLOR_BGR2HSV);

        // 2. Threshold for Skin Color (Generic values, might need tweaking based on lighting)
        // Lower: (0, 20, 70), Upper: (20, 255, 255) covers most skin tones
        Scalar lowerSkin(0, 20, 70);
        Scalar upperSkin(20, 255, 255);
        inRange(hsv, lowerSkin, upperSkin, mask);

        // 3. Clean up noise (Erosion/Dilation)
        erode(mask, mask, Mat(), Point(-1, -1), 2);
        dilate(mask, mask, Mat(), Point(-1, -1), 2);
        GaussianBlur(mask, mask, Size(5, 5), 0);

        // 4. Find Contours
        vector<vector<Point>> contours;
        findContours(mask, contours, RETR_TREE, CHAIN_APPROX_SIMPLE);

        int count = 0;

        if (!contours.empty()) {
            // Find largest contour (assumed to be the hand)
            size_t maxIdx = 0;
            double maxArea = 0;
            for (size_t i = 0; i < contours.size(); i++) {
                double area = contourArea(contours[i]);
                if (area > maxArea) {
                    maxArea = area;
                    maxIdx = i;
                }
            }

            // Only process if the hand is big enough
            if (maxArea > 3000) {
                vector<Point> maxContour = contours[maxIdx];

                // Convex Hull
                vector<int> hullIndices;
                convexHull(maxContour, hullIndices, false);

                // Convexity Defects (The gaps between fingers)
                vector<Vec4i> defects;
                if (hullIndices.size() > 3) {
                    convexityDefects(maxContour, hullIndices, defects);

                    for (const auto& v : defects) {
                        float depth = (float)v[3] / 256.0f;
                        if (depth > 10) { // Filter shallow defects (noise)
                            int startIdx = v[0];
                            int endIdx = v[1];
                            int farIdx = v[2];

                            Point pStart = maxContour[startIdx];
                            Point pEnd = maxContour[endIdx];
                            Point pFar = maxContour[farIdx];

                            // Cosine Law to check angle (fingers are usually sharp angles)
                            double a = norm(pEnd - pStart);
                            double b = norm(pFar - pStart);
                            double c = norm(pFar - pEnd);
                            double angle = acos((b * b + c * c - a * a) / (2 * b * c)) * 180 / CV_PI;

                            // If angle is acute (< 90), it's likely a finger gap
                            if (angle <= 90) {
                                count++;
                            }
                        }
                    }
                }
                // Logic: 0 defects = 1 finger (pointing) or fist.
                // Let's assume 1 finger minimum if area is large.
                // Formula: Fingers = Gaps + 1
                detectedFingers = count + 1;

                // Cap at 5
                if (detectedFingers > 5) detectedFingers = 5;
            }
            else {
                detectedFingers = 0; // Hand not found
            }
        }
        else {
            detectedFingers = 0;
        }

        // 5. Stability Logic (Must hold gesture to trigger)
        if (detectedFingers == lastStableCount && detectedFingers > 0) {
            holdTime += dt;
            if (holdTime >= REQUIRED_HOLD_TIME) {
                triggerAction = true;
                // Draw Green text indicating locked
                putText(frame, "LOCKED: " + to_string(detectedFingers), Point(50, 40), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);
            }
            else {
                triggerAction = false;
                // Draw Yellow text indicating loading
                putText(frame, "Hold: " + to_string(detectedFingers), Point(50, 40), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 255), 2);
            }
        }
        else {
            lastStableCount = detectedFingers;
            holdTime = 0;
            triggerAction = false;
            line(frame, Point(50, 80), Point(50 + (holdTime / REQUIRED_HOLD_TIME) * 200, 80), Scalar(0, 255, 255), 5);
            putText(frame, "Detecting...", Point(50, 40), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);
        }

        // Show the camera view in a separate small window
        imshow("Gesture Control", frame);
        isWindowOpen = true;
    }

    // Returns true ONE time when the gesture is locked, then resets
    bool consumeTrigger(int& outFingerCount) {
        if (triggerAction) {
            outFingerCount = lastStableCount;
            // Reset to prevent machine-gun triggering
            holdTime = 0;
            triggerAction = false;
            return true;
        }
        return false;
    }
};




//Rounded Corner Buttons
class RoundedRectangleShape : public Shape { // Taking colors,textures,etc. from 'Shape'
public:
    RoundedRectangleShape(const Vector2f& size = Vector2f(0, 0), float radius = 0, unsigned int cornerPointCount = 10)
        : m_size(size), m_radius(radius), m_cornerPointCount(cornerPointCount) {
        update();
    }
    //
    void setSize(const Vector2f& size) {
        m_size = size;
        update();
    }
    const Vector2f& getSize() const { return m_size; }
    //---------------------------------------------------
    void setCornersRadius(float radius) {
        m_radius = radius; update();
    }
    float getCornersRadius() const { return m_radius; }
    //---------------------------------------------------
    void setCornerPointCount(unsigned int count) {   //For drawing the corner, checks how many dots make the corner
        m_cornerPointCount = count; update();
    }
    virtual size_t getPointCount() const {
        return m_cornerPointCount * 4;
    }
    //---------------------------------------------------
    virtual Vector2f getPoint(size_t index) const { // Checks the co-ordinates of all points to draw
        if (index >= m_cornerPointCount * 4)
            return Vector2f(0, 0);
        float deltaAngle = 90.0f / (m_cornerPointCount - 1);
        Vector2f center;
        unsigned int centerIndex = index / m_cornerPointCount; // Figures out which corner to draw
        static const float pi = 3.141592654f;
        switch (centerIndex) { // For center point of the virtual circle to draw the rounded corner
        case 0: center.x = m_size.x - m_radius; center.y = m_radius; break;
        case 1: center.x = m_radius; center.y = m_radius; break;
        case 2: center.x = m_radius; center.y = m_size.y - m_radius; break;
        case 3: center.x = m_size.x - m_radius; center.y = m_size.y - m_radius; break;
        }
        // Math to get the points of the circle
        return Vector2f(m_radius * cos(deltaAngle * (index - centerIndex * m_cornerPointCount) * pi / 180 + centerIndex * pi / 2) + center.x,
            -m_radius * sin(deltaAngle * (index - centerIndex * m_cornerPointCount) * pi / 180 + centerIndex * pi / 2) + center.y);
    }
private:
    Vector2f m_size; // Width and Height
    float m_radius; // How round the corners
    unsigned int m_cornerPointCount; // Smoothness of Roundness
};

//Gesture Tracking


// Buttons
class OptionButton {
public:
    RoundedRectangleShape shape;
    Text text; // Options Text
    Text prefix; // Options Number (A. B. C. D.)
    Color baseFillColor = UI_BASE_COLOR; // The Backgorund of Buttons
    Color baseOutlineColor = DEFAULT_OUTLINE_COLOR; // The Border of Buttons
    Vector2f originalPos;
    OptionButton(float x, float y, float w, float h, const string& prefixText, const Font& font);
    void update(Vector2i mousePos);
    void setOptionText(const string& optionText);
    void setColor(const Color& color) { shape.setFillColor(color); shape.setOutlineColor(color); }
    void resetColor() { shape.setFillColor(baseFillColor); shape.setOutlineColor(baseOutlineColor); }
    bool isClicked(Vector2i mousePos) const {
        return shape.getGlobalBounds().contains(static_cast<Vector2f>(mousePos));
    }
    void draw(RenderWindow& window) const;
    void setPosition(const Vector2f& pos);
};

// Function Declarations
int getHighScore();
void saveHighScore(int currentScore);
void spawnParticles(vector<Particle>& particles, Vector2f pos, Color color);
vector<QuizQuestion> loadQuestionsFromFile(const string& filename);



/*--------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------
----------------------------------------  Main Fucntion  -------------------------------------------
----------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------*/


int main() {
    //Rendering Window
    RenderWindow window(VideoMode({ WINDOW_WIDTH, WINDOW_HEIGHT }), "C++ Logic Builder");
    window.setFramerateLimit(60);
    srand(static_cast<unsigned>(time(0)));
    GestureTracker gestureTracker;

    //Streak on correct Answers
    int comboStreak = 0;

    //ScreenShake on incorrect Answers
    View originalView = window.getDefaultView();
    View shakeView = originalView;
    float shakeTime = 0.0f; // Initial Shaking Time
    float shakeMagnitude = 10.0f; // Shake Magnitude Intensity

    // Audio Settings
    bool musicEnabled = true;
    bool sfxEnabled = true;
    float musicVolume = 50.0f;
    bool cameraEnabled = true;

    // Load Resources
    
    Font uifont; // UI Font
    if (!uifont.openFromFile("Montserrat.ttf")) { // Loading Font Montserrat
        cerr << "Error: Could not load arial.ttf." << endl;
        return -1;
    }
    Font codeFont; //Code Font
    if (!codeFont.openFromFile("JetBrainsMono.ttf")) { // Loading Code Question Font JetBrains Mono
        codeFont = uifont;
    }

    Font titleFont;
    if (!titleFont.openFromFile("Orbitron.ttf")) { // Loading Title (Easy mode, etc.) Font Orbitron
        titleFont = uifont;
    }
    Texture backgroundTexture;
    unique_ptr<Sprite> backgroundSprite; // Smart pointer to kill the bg img when it dies/ Saves memory
    if (backgroundTexture.loadFromFile("download.jpg")) { // Loading the background image
        backgroundSprite = make_unique<Sprite>(backgroundTexture); // Building canvas
        const Vector2u textureSize = backgroundTexture.getSize(); // For exact dimesnions loading of the image
        float scaleX = (float)WINDOW_WIDTH / textureSize.x; // Fits the image
        float scaleY = (float)WINDOW_HEIGHT / textureSize.y; // Fits the image
        float scale = max(scaleX, scaleY); // For best Orientation
        backgroundSprite->setScale({ scale, scale }); // Applying scale
        float offsetX = (WINDOW_WIDTH - backgroundSprite->getGlobalBounds().size.x) / 2.0f; // Centering Horizontally
        float offsetY = (WINDOW_HEIGHT - backgroundSprite->getGlobalBounds().size.y) / 2.0f; // Centering Veritcally
        backgroundSprite->setPosition({ offsetX, offsetY }); // Applying the position of centering
    }

    // Audio Loading
    SoundBuffer correctBuffer, incorrectBuffer;
    Sound correctSound(correctBuffer), incorrectSound(incorrectBuffer);
    bool hasSound = true;
    if (!correctBuffer.loadFromFile("correct.wav")) hasSound = false; else correctSound.setBuffer(correctBuffer); // Correct answer sound
    if (!incorrectBuffer.loadFromFile("fail.wav")) hasSound = false; else incorrectSound.setBuffer(incorrectBuffer); // Incorrect answer buzzer

    Music bgMusic;
    if (bgMusic.openFromFile("bgmusic.ogg")) { // Load background Music
        bgMusic.setLooping(true); // Loop forever
        bgMusic.setVolume(50.f); // Volume Setting
        bgMusic.play(); // For playing the background music
    }
    else {
        cerr << "Warning: Background music failed to load." << endl;
    }

    // Global State Variables
    vector<QuizQuestion> allQuestions;  // Holds all the questions from all files
    unsigned int currentQuestionIndex = 0;
    GameState currentState = MENU;
    int score = 0;
    int totalQuestions = 0;
    int actualTotalQuestions = 0;

    float timeLeft = TIME_PER_QUESTION;
    Clock dtClock; // Checks time since last frame was drawn to help the timerBar work correctly
    Clock effectClock; // Used for Background Pulse Effect
    RoundedRectangleShape timerTrack({ (float)WINDOW_WIDTH - 100.f, 20.f }, 10.f, 10);
    timerTrack.setPosition({ 50.f, 10.f }); // Centered with 50px padding on sides
    timerTrack.setFillColor(Color(20, 20, 20, 150)); // Dark Gray, semi-transparent
    timerTrack.setOutlineThickness(2);
    timerTrack.setOutlineColor(Color(100, 100, 255, 100)); // Faint Blue glow
    RoundedRectangleShape timerBar({ (float)WINDOW_WIDTH - 100.f, 20.f }, 10.f, 10);
    timerBar.setPosition({ 50.f, 10.f }); // Same position as track
    timerBar.setFillColor(Color::Green);
    //RectangleShape timerBar;  // Timer Bar on the top
    //timerBar.setPosition({ 0,0 });
    //timerBar.setSize({ (float)WINDOW_WIDTH, 10.f });
    //timerBar.setFillColor(Color::Green); // Timer Bar color in the beginning

    vector<Particle> particles;
    vector<FloatingText> floatTexts;

    // Custom Input Variables
    bool isTypingCustomAmount = false;
    string customInputString = "";
    Text customInputDisplay(codeFont);
    customInputDisplay.setCharacterSize(40);
    customInputDisplay.setFillColor(Color(180, 200, 255));

    bool isAnswerLocked = false;
    bool autoNext = false;
    Clock feedbackTimer;
    const Time feedbackDuration = seconds(1.5f); // Waits for 1.5 seconds before going to next question
    string currentDifficultyName = ""; // To display "Hard Mode", etc.

    /*--------------------------------------------  UI -----------------------------------------------*/

    Text titleText(titleFont);
    titleText.setString("C++ Logic Builder");
    titleText.setCharacterSize(48);
    titleText.setFillColor(Color(180, 200, 255)); // Bluish Color (Sky Blue)
    FloatRect tr = titleText.getLocalBounds(); // Gets the length/picels of the text
    titleText.setOrigin({ tr.position.x + tr.size.x / 2.0f, 0 }); // Move the origin to center
    titleText.setPosition({ WINDOW_WIDTH / 2.0f, 60.0f });

    // Beginning
    Text scoreText(uifont);
    scoreText.setString("Score: 0");
    scoreText.setCharacterSize(24);
    scoreText.setPosition({ 10, WINDOW_HEIGHT - 40 }); // Shows the score on bottom
    scoreText.setFillColor(Color::White);

    Text questionText(codeFont);
    questionText.setString("Question text");
    questionText.setCharacterSize(28);
    questionText.setFillColor(Color::Yellow);
    questionText.setLineSpacing(1.5f);

    /*--------------------------------------------------------------------------------------------------
    -------------------------------------------  BUTTONS  ----------------------------------------------
    --------------------------------------------------------------------------------------------------*/

    // Main Menu
    OptionButton startBtn(WINDOW_WIDTH / 2 - 120, 300, 240, 60, "", uifont);
    OptionButton settingsBtn(WINDOW_WIDTH / 2 - 120, 380, 240, 60, "", uifont);
    OptionButton exitBtn(WINDOW_WIDTH / 2 - 120, 460, 240, 60, "", uifont);
    startBtn.setOptionText("Start Game");
    settingsBtn.setOptionText("Settings");
    exitBtn.setOptionText("Exit Game");

    // Settings Buttons
    OptionButton toggleMusicBtn(WINDOW_WIDTH / 2 - 150, 250, 300, 60, "", uifont); //Music Button (On/Off)
    OptionButton toggleSfxBtn(WINDOW_WIDTH / 2 - 150, 330, 300, 60, "", uifont);  //SFX Button (On/Off)
    OptionButton toggleCamBtn(WINDOW_WIDTH / 2 - 150, 410, 300, 60, "", uifont); // Adjusted Y pos
    

    // Adjust Volume Buttons
    OptionButton volDownBtn(WINDOW_WIDTH / 2 - 150, 490, 60, 60, "", uifont);
    OptionButton volUpBtn(WINDOW_WIDTH / 2 + 90, 490, 60, 60, "", uifont);
    OptionButton backSettingsBtn(WINDOW_WIDTH / 2 - 120, 580, 240, 60, "", uifont);
    

    toggleMusicBtn.setOptionText("Music: ON");
    toggleMusicBtn.baseFillColor = Color(40, 100, 40, 200);  // Dark Green but transparent
    toggleMusicBtn.resetColor();

    toggleSfxBtn.setOptionText("SFX: ON");
    toggleSfxBtn.baseFillColor = Color(40, 100, 40, 200); // Dark Green but transparent
    toggleSfxBtn.resetColor();

    toggleCamBtn.setOptionText("Camera: ON");
    toggleCamBtn.baseFillColor = Color(40, 100, 40, 200);
    toggleCamBtn.resetColor();

    volDownBtn.setOptionText("-");
    volUpBtn.setOptionText("+");
    backSettingsBtn.setOptionText("Back");
    backSettingsBtn.baseFillColor = Color(150, 50, 50, 200); // Dark Red but transparent
    backSettingsBtn.resetColor();

    // Text to display volume number
    Text volumeDisplay(uifont, "Vol: 50", 30);
    volumeDisplay.setFillColor(Color::White);


    // DIFFICULTY SELECT
    const float DIFF_BTN_Y = 300.0f;
    OptionButton easyBtn(WINDOW_WIDTH / 2 - 120, DIFF_BTN_Y, 240, 60, "", uifont);
    OptionButton mediumBtn(WINDOW_WIDTH / 2 - 120, DIFF_BTN_Y + 80, 240, 60, "", uifont);
    OptionButton hardBtn(WINDOW_WIDTH / 2 - 120, DIFF_BTN_Y + 160, 240, 60, "", uifont);

    easyBtn.setOptionText("Easy");
    easyBtn.baseFillColor = Color(50, 150, 50, 200); // Greenish Color
    easyBtn.resetColor();

    mediumBtn.setOptionText("Medium");
    mediumBtn.baseFillColor = Color(200, 150, 50, 200); // Orangish Color
    mediumBtn.resetColor();

    hardBtn.setOptionText("Hard");
    hardBtn.baseFillColor = Color(150, 50, 50, 200); // Redish Color
    hardBtn.resetColor();

    // Limit Select
    OptionButton customLimitBtn(WINDOW_WIDTH / 2 - 150, 300, 300, 60, "", uifont);
    customLimitBtn.setOptionText("Enter Desired Questions");
    customLimitBtn.baseFillColor = Color(100, 50, 100, 200);
    customLimitBtn.resetColor();

    OptionButton confirmLimitBtn(WINDOW_WIDTH / 2 + 160, 300, 100, 60, "", uifont);
    confirmLimitBtn.setOptionText("ENTER");
    confirmLimitBtn.baseFillColor = Color(50, 150, 50, 200); // Medium Dark Green
    confirmLimitBtn.resetColor();

    OptionButton limitAllBtn(WINDOW_WIDTH / 2 - 150, 440, 300, 60, "", uifont);
    limitAllBtn.setOptionText("Play All");
    limitAllBtn.baseFillColor = Color(50, 100, 50, 200); // Dark Green
    limitAllBtn.resetColor();

    // In Game
    OptionButton pauseBtn(WINDOW_WIDTH - 140, 50, 130, 40, "", uifont);
    pauseBtn.setOptionText("Pause (Esc)");
    OptionButton skipBtn(WINDOW_WIDTH - 300, 50, 150, 40, "", uifont);
    skipBtn.setOptionText("Next (Right)");
    OptionButton backBtn(WINDOW_WIDTH - 460, 50, 150, 40, "", uifont);
    backBtn.setOptionText("Prev (Left)");
    OptionButton endQuizBtn(WINDOW_WIDTH / 2 - 120, 460, 240, 60, "", uifont);
    endQuizBtn.setOptionText("End Quiz");
    endQuizBtn.baseFillColor = Color(150, 50, 50, 200); // Reddish Color
    endQuizBtn.resetColor();
    OptionButton quizCamBtn(20, 50, 150, 40, "", uifont); // Camera Toggling
    if (cameraEnabled) {
        quizCamBtn.setOptionText("Cam: ON");
        quizCamBtn.baseFillColor = Color(40, 100, 40, 200); // Green
    }
    else {
        quizCamBtn.setOptionText("Cam: OFF");
        quizCamBtn.baseFillColor = Color(150, 40, 40, 200); // Red
    }
    quizCamBtn.resetColor();

    // Answer Options
    const float OPTION_WIDTH = WINDOW_WIDTH / 2.0f - 100;
    const float OPTION_HEIGHT = 80;
    const float START_Y = WINDOW_HEIGHT - 250;
    const float PADDING = 100;

    OptionButton options[4] = {
       OptionButton(50, START_Y, OPTION_WIDTH, OPTION_HEIGHT, "A:", uifont),
       OptionButton(WINDOW_WIDTH / 2.0f + 50, START_Y, OPTION_WIDTH, OPTION_HEIGHT,"B:", uifont),
       OptionButton(50, START_Y + PADDING, OPTION_WIDTH, OPTION_HEIGHT,"C:", uifont),
       OptionButton(WINDOW_WIDTH / 2.0f + 50, START_Y + PADDING, OPTION_WIDTH, OPTION_HEIGHT, "D:", uifont)
    };

    /*--------------------------------------------------------------------------------------------------
    -----------------------------------------  BUTTONS END  --------------------------------------------
    --------------------------------------------------------------------------------------------------*/

    /*-----------------------------------------   UI END  --------------------------------------------*/


    // Helpers
    auto loadQuestion = [&]() { //[&] is the Capture List. It allows the fucntion to see and modify variables decaled outside
        if (currentQuestionIndex < actualTotalQuestions) {
            const auto& q = allQuestions[currentQuestionIndex];
            questionText.setString(q.questionText);
            // Text Positioning
            FloatRect textBounds = questionText.getLocalBounds(); // Center the text
            questionText.setOrigin({ textBounds.position.x + textBounds.size.x / 2.0f, textBounds.position.y + textBounds.size.y / 2.0f });
            questionText.setPosition({ WINDOW_WIDTH / 2.0f, WINDOW_HEIGHT / 2.8f });
            // Update the Buttons for the next question
            for (int i = 0; i < 4; ++i) {
                options[i].setOptionText(q.options[i]);
                options[i].resetColor();
            }
            // Checks if the question has been answered
            if (q.userSelectedOption != -1) {
                isAnswerLocked = true;
                options[q.correctAnswerIndex].setColor(CORRECT_COLOR); // Green
                if (q.userSelectedOption != q.correctAnswerIndex)
                    options[q.userSelectedOption].setColor(INCORRECT_COLOR); // Red
                timeLeft = 0;
                timerBar.setSize({ 0, 20.f });
                autoNext = false;
            }
            else {
                isAnswerLocked = false; // Unlock
                timeLeft = TIME_PER_QUESTION; // Reset Timer
                timerBar.setSize({ (float)WINDOW_WIDTH - 100.f, 20.f });
                timerBar.setFillColor(Color::Green);
                autoNext = false;
            }
            //// Update the timer for next question
            //timeLeft = TIME_PER_QUESTION;
            //timerBar.setSize({ (float)WINDOW_WIDTH - 100.f, 20.f });
            //timerBar.setFillColor(Color::Green);
            //isAnswerLocked = false;
        }
        else {
            currentState = GAME_OVER;
        }
        };

    // Starts Game
    auto startGame = [&](int limit) {
        score = 0;
        currentQuestionIndex = 0;
        for (auto& q : allQuestions) { // Reset the memory for all questions
            q.userSelectedOption = -1;
        }
        actualTotalQuestions = min(limit, totalQuestions); // Limit is the number of questions user wants, it checks whether user's number is smaller than the actual present questions and chooses the min questions to save game from crashing
        random_device rd; // Shuffling Seed
        mt19937 g(rd()); // Randomizing generator for questions
        shuffle(allQuestions.begin(), allQuestions.end(), g); // Shuffles all the questions
        loadQuestion();
        currentState = QUIZ_MODE;
        isTypingCustomAmount = false;
        customInputString = "";
        };

    // Switches files
    auto selectDifficulty = [&](string filename, string displayName) {
        allQuestions = loadQuestionsFromFile(filename);
        if (allQuestions.empty()) {
            cout << "Could not find " << filename << ", trying fallback 'questions.txt'..." << endl;
            allQuestions = loadQuestionsFromFile("questions.txt");
        }

        if (allQuestions.empty()) { // All Files empty
            cerr << "CRITICAL: No questions found!" << endl;
            currentState = MENU;
            return;
        }
        totalQuestions = static_cast<int>(allQuestions.size());
        limitAllBtn.setOptionText("Play All (" + to_string(totalQuestions) + ")");
        currentDifficultyName = displayName;
        currentState = SET_LIMIT;
        };

    // Fade transitions
    RectangleShape fadeRect({ (float)WINDOW_WIDTH, (float)WINDOW_HEIGHT });
    fadeRect.setFillColor(Color::Black); // Starts fully black
    float fadeAlpha = 255.0f;

    // Lambda Function to trigger a flash
    auto triggerFade = [&]() { fadeAlpha = 255.0f; };

    // Main Game Loop
    while (window.isOpen()) {
        // Calculate Delta Time (dt)
        Time dtTime = dtClock.restart();
        float dt = dtTime.asSeconds();

        gestureTracker.update(dt, (currentState == QUIZ_MODE));
        int gestureFingers = 0;
        bool gestureTriggered = gestureTracker.consumeTrigger(gestureFingers);

        // Calculate Background Pulse (Background Continuous Color Changing
        Time elapsed = effectClock.getElapsedTime();
        float wave = (sin(elapsed.asSeconds() * 1.0f) + 1.0f) / 2.0f; // Divides the number into a range from 0 to 1 for smooth effects
        uint8_t r = static_cast<uint8_t>(200 + (wave * 55)); // calculates the red component
        uint8_t g = static_cast<uint8_t>(200); // calculates the green component
        uint8_t b = 255; // calculates the blue component
        // uint8_t forces number to be within 0-255
        Color animatedBgColor(r, g, b); // Makes the rgb color


        /*-----------------------------------------   Event Pollings  --------------------------------------------*/

        while (const optional event = window.pollEvent()) { // Checks for Keyboard Input
            if (event->is<Event::Closed>()) { window.close(); } // Checks for the closing 'X' click on the windows title bar
            // Text Entry
            if (currentState == SET_LIMIT && isTypingCustomAmount) { // User entered the button to type on set limit menu
                if (const auto* textEvent = event->getIf<Event::TextEntered>()) { // Checks for the entered TExt
                    uint32_t unicode = textEvent->unicode; // uint32_t is 32-btis unisgned integer, converts the text into unicode
                    if (unicode >= '0' && unicode <= '9') { // Only lets numbers through the entry button
                        if (customInputString.length() < 4) customInputString += static_cast<char>(unicode); //Checks if the length of characters is less than 4 and adds to the string variable on screen
                    }
                    else if (unicode == 8 && !customInputString.empty()) customInputString.pop_back(); // 8 is for backspace, deletes the last character form left
                    else if (unicode == 13 && !customInputString.empty()) { // It is for Enter Key unicode
                        try { // try is used to prevent conversion fails
                            int val = stoi(customInputString); // stoi converts from str to int
                            if (val > 0) startGame(val); // If the entered amount is grater than 0, start game
                        }
                        catch (...) {} // if aboe block crashes for some reason, this block gets all the errors so the game doesn't close abruptly
                    }
                }
            }
            // Key Presses
            if (const auto* keyEvent = event->getIf<Event::KeyPressed>()) { // Checks for keys being pressed
                if (keyEvent->code == Keyboard::Key::Escape) { // Escape key is pressed
                    if (currentState == QUIZ_MODE) currentState = PAUSED; // Pauses the game
                    else if (currentState == PAUSED) currentState = QUIZ_MODE; // Resumes the game
                    else if (currentState == SELECT_DIFFICULTY) currentState = MENU; // Returns to menu
                    else if (currentState == SETTINGS) currentState = MENU;
                    else if (currentState == MENU) window.close();
                    else if (currentState == SET_LIMIT) {
                        if (isTypingCustomAmount) { isTypingCustomAmount = false; customLimitBtn.resetColor(); } // resets all the text entered in the "Enter the desired questions"
                        else currentState = SELECT_DIFFICULTY; // If no text is entered it, comes back to  Difficulty Selection Tab
                    }
                }
                else if (currentState == QUIZ_MODE) {
                    if (keyEvent->code == Keyboard::Key::Right) {
                        currentQuestionIndex++; // Skips to the next Question
                        loadQuestion();
                    }
                    else if (keyEvent->code == Keyboard::Key::Left) {
                        if (currentQuestionIndex > 0) currentQuestionIndex--; // Comes back to the previous Question
                        else if (currentQuestionIndex == 0) currentQuestionIndex = 0;
                        loadQuestion();
                    }
                }
                else if (currentState == SETTINGS) {
                    if (keyEvent->code == Keyboard::Key::Right) { // Volume Increase
                        musicVolume = min(100.0f, musicVolume + 1.0f);
                        bgMusic.setVolume(musicVolume);
                    }
                    else if (keyEvent->code == Keyboard::Key::Left) { // Voilume Decrease
                        musicVolume = max(0.0f, musicVolume - 1.0f);
                        bgMusic.setVolume(musicVolume);
                    }
                }
            }

            // Mouse Clicks
            else if (const auto* mouseEvent = event->getIf<Event::MouseButtonPressed>()) {
                if (mouseEvent->button == Mouse::Button::Left) {
                    Vector2i mousePos = mouseEvent->position;

                    if (currentState == MENU) {
                        if (startBtn.isClicked(mousePos)) {
                            triggerFade(); // Fade Flashes on the screen
                            currentState = SELECT_DIFFICULTY; // Goes to DIfficulty Selction
                        }
                        if (settingsBtn.isClicked(mousePos)) {
                            triggerFade(); // Fade Flases on the screen
                            currentState = SETTINGS; // Goes to Settings
                        }
                        if (exitBtn.isClicked(mousePos)) window.close(); // Closes the game
                    }
                    else if (currentState == SETTINGS) {
                        if (backSettingsBtn.isClicked(mousePos)) {
                            currentState = MENU; // Returns to Menu
                        }

                        // Toggle Music
                        if (toggleMusicBtn.isClicked(mousePos)) {
                            musicEnabled = !musicEnabled;
                            if (musicEnabled) {
                                toggleMusicBtn.setOptionText("Music: ON");
                                toggleMusicBtn.baseFillColor = Color(40, 100, 40, 200); // Green
                                bgMusic.play();
                            }
                            else {
                                toggleMusicBtn.setOptionText("Music: OFF");
                                toggleMusicBtn.baseFillColor = Color(150, 40, 40, 200); // Red
                                bgMusic.pause();
                            }
                            toggleMusicBtn.resetColor();
                        }
                        // Toggle SFX
                        if (toggleSfxBtn.isClicked(mousePos)) {
                            sfxEnabled = !sfxEnabled;
                            if (sfxEnabled) {
                                toggleSfxBtn.setOptionText("SFX: ON");
                                toggleSfxBtn.baseFillColor = Color(40, 100, 40, 200); // Green
                            }
                            else {
                                toggleSfxBtn.setOptionText("SFX: OFF");
                                toggleSfxBtn.baseFillColor = Color(150, 40, 40, 200); // Red
                            }
                            toggleSfxBtn.resetColor();
                        }
                        //Toggle Camera
                        if (toggleCamBtn.isClicked(mousePos)) {
                            cameraEnabled = !cameraEnabled;
                            gestureTracker.setEnabled(cameraEnabled); // Physically turn on/off
                            if (cameraEnabled) {
                                toggleCamBtn.setOptionText("Camera: ON");
                                toggleCamBtn.baseFillColor = Color(40, 100, 40, 200);
                            }
                            else {
                                toggleCamBtn.setOptionText("Camera: OFF");
                                toggleCamBtn.baseFillColor = Color(150, 40, 40, 200);
                            }
                            toggleCamBtn.resetColor();
                        }
                        // Volume Controls
                        if (volUpBtn.isClicked(mousePos)) {
                            musicVolume = min(100.0f, musicVolume + 10.0f);
                            bgMusic.setVolume(musicVolume);
                        }
                        if (volDownBtn.isClicked(mousePos)) {
                            musicVolume = max(0.0f, musicVolume - 10.0f);
                            bgMusic.setVolume(musicVolume);
                        }
                    }
                    else if (currentState == SELECT_DIFFICULTY) {
                        if (easyBtn.isClicked(mousePos)) {
                            triggerFade();
                            selectDifficulty("easy.txt", "Easy Mode");
                        }
                        else if (mediumBtn.isClicked(mousePos)) {
                            triggerFade();
                            selectDifficulty("medium.txt", "Medium Mode");
                        }
                        else if (hardBtn.isClicked(mousePos)) {
                            triggerFade();
                            selectDifficulty("hard.txt", "Hard Mode");
                        }
                    }
                    else if (currentState == SET_LIMIT) {
                        if (customLimitBtn.isClicked(mousePos)) {
                            isTypingCustomAmount = true;
                            customInputString = "";
                        }
                        else if (confirmLimitBtn.isClicked(mousePos) && !customInputString.empty()) {
                            try {
                                int val = stoi(customInputString);
                                if (val > 0) {
                                    triggerFade();
                                    startGame(val);
                                }
                            }
                            catch (...) {}
                        }
                        else if (limitAllBtn.isClicked(mousePos)) {
                            triggerFade();
                            startGame(totalQuestions);
                        }
                        else { isTypingCustomAmount = false; customLimitBtn.resetColor(); }
                    }
                    else if (currentState == PAUSED) {
                        if (startBtn.isClicked(mousePos)) currentState = QUIZ_MODE;
                        if (endQuizBtn.isClicked(mousePos)) {
                            triggerFade();
                            currentState = GAME_OVER;
                        }
                        if (exitBtn.isClicked(mousePos)) currentState = MENU;
                    }
                    else if (currentState == GAME_OVER) {
                        if (startBtn.isClicked(mousePos)) currentState = MENU;
                        if (exitBtn.isClicked(mousePos)) window.close();
                    }
                    else if (currentState == QUIZ_MODE) {
                        if (quizCamBtn.isClicked(mousePos)) {
                            cameraEnabled = !cameraEnabled;
                            gestureTracker.setEnabled(cameraEnabled); // Physically toggle

                            // Update Visuals for this button
                            if (cameraEnabled) {
                                quizCamBtn.setOptionText("Cam: ON");
                                quizCamBtn.baseFillColor = Color(40, 100, 40, 200);
                            }
                            else {
                                quizCamBtn.setOptionText("Cam: OFF");
                                quizCamBtn.baseFillColor = Color(150, 40, 40, 200);
                            }
                            quizCamBtn.resetColor();

                            // Update the Settings menu button too (to keep them synced)
                            if (cameraEnabled) {
                                toggleCamBtn.setOptionText("Camera: ON");
                                toggleCamBtn.baseFillColor = Color(40, 100, 40, 200);
                            }
                            else {
                                toggleCamBtn.setOptionText("Camera: OFF");
                                toggleCamBtn.baseFillColor = Color(150, 40, 40, 200);
                            }
                            toggleCamBtn.resetColor();
                        }
                        if (pauseBtn.isClicked(mousePos)) {
                            currentState = PAUSED;
                            continue;
                        }
                        if (skipBtn.isClicked(mousePos)) {
                            currentQuestionIndex++;
                            loadQuestion();
                            continue;
                        }
                        if (backBtn.isClicked(mousePos)) {
                            if (currentQuestionIndex > 0)
                                currentQuestionIndex--;
                            else if (currentQuestionIndex == 0)
                                currentQuestionIndex = 0;
                            loadQuestion();
                            continue;
                        }
                        if (!isAnswerLocked) { // Preventss from selecting two options
                            for (int i = 0; i < 4; ++i) {
                                if (options[i].isClicked(mousePos)) {
                                    allQuestions[currentQuestionIndex].userSelectedOption = i; // Saves the selection to the memory
                                    const auto& currentQ = allQuestions[currentQuestionIndex];
                                    if (i == currentQ.correctAnswerIndex) { // Correct Answer
                                        score++;
                                        comboStreak++;
                                        float pitch = min(2.0f, 1.0f + (comboStreak * 0.1f)); // Pitch of Ding increases
                                        correctSound.setPitch(pitch);
                                        options[i].setColor(CORRECT_COLOR); // Turns the button Green
                                        if (hasSound && sfxEnabled) correctSound.play(); // Play the correct Sound
                                        floatTexts.emplace_back(uifont, "+1", static_cast<float>(mousePos.x), static_cast<float>(mousePos.y - 40)); // Shows the Floating Text
                                    }
                                    else { /// For incorrect answers
                                        options[i].setColor(INCORRECT_COLOR); // Turns the selected wrong answer Red
                                        options[currentQ.correctAnswerIndex].setColor(CORRECT_COLOR); // Turns the correct option Green
                                        Vector2f center = options[i].shape.getPosition() + (options[i].shape.getSize() / 2.f); // Spawns the 20 particle on the incorrect answer
                                        spawnParticles(particles, center, Color::Red); // Particles color to Red
                                        shakeTime = 0.5f; // Screen Shake Time
                                        comboStreak = 0; // Resets the combo Streak
                                        correctSound.setPitch(1.0f); // Resets the Pitch
                                        if (hasSound && sfxEnabled) incorrectSound.play(); // Plays the incorrect buzzer sound
                                    }
                                    isAnswerLocked = true; // Stops user from clicking anything
                                    autoNext = true;
                                    feedbackTimer.restart(); // Restarts the timer
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }

        /*-----------------------------------------   Event Polling End  --------------------------------------------*/

        // --- NEW CODE: HANDLE GESTURE INPUTS ---
        if (gestureTriggered) {
            cout << "Gesture Triggered: " << gestureFingers << endl; // Debugging

            // 5 FINGERS: PAUSE / RESUME logic
            if (gestureFingers == 5) {
                if (currentState == QUIZ_MODE) currentState = PAUSED;
                else if (currentState == PAUSED) currentState = QUIZ_MODE;
            }
            // 1-4 FINGERS: SELECT ANSWER (Only in Quiz Mode)
            else if (currentState == QUIZ_MODE && !isAnswerLocked) {
                int selectedIndex = -1;

                if (gestureFingers == 1) selectedIndex = 0; // Option A
                else if (gestureFingers == 2) selectedIndex = 1; // Option B
                else if (gestureFingers == 3) selectedIndex = 2; // Option C
                else if (gestureFingers == 4) selectedIndex = 3; // Option D

                // Logic copied from your Mouse Click event
                if (selectedIndex != -1) {
                    allQuestions[currentQuestionIndex].userSelectedOption = selectedIndex;
                    const auto& currentQ = allQuestions[currentQuestionIndex];

                    if (selectedIndex == currentQ.correctAnswerIndex) {
                        // Correct
                        score++;
                        comboStreak++;
                        float pitch = min(2.0f, 1.0f + (comboStreak * 0.1f));
                        correctSound.setPitch(pitch);
                        options[selectedIndex].setColor(CORRECT_COLOR);
                        if (hasSound && sfxEnabled) correctSound.play();
                        // Floating text math...
                        floatTexts.emplace_back(uifont, "+1", WINDOW_WIDTH / 2, WINDOW_HEIGHT / 2);
                    }
                    else {
                        // Incorrect
                        options[selectedIndex].setColor(INCORRECT_COLOR);
                        options[currentQ.correctAnswerIndex].setColor(CORRECT_COLOR);
                        Vector2f center = options[selectedIndex].shape.getPosition() + (options[selectedIndex].shape.getSize() / 2.f);
                        spawnParticles(particles, center, Color::Red);
                        shakeTime = 0.5f;
                        comboStreak = 0;
                        correctSound.setPitch(1.0f);
                        if (hasSound && sfxEnabled) incorrectSound.play();
                    }
                    isAnswerLocked = true;
                    autoNext = true;
                    feedbackTimer.restart();
                }
            }
        }

        // Update Game Logic

        Vector2i mPos = Mouse::getPosition(window); // Gets mouse Position

        // Hover Effects
        if (currentState == MENU) {
            startBtn.update(mPos);
            settingsBtn.update(mPos);
            exitBtn.update(mPos);
        }
        else if (currentState == SETTINGS) {
            toggleMusicBtn.update(mPos);
            toggleSfxBtn.update(mPos);
            toggleCamBtn.update(mPos);
            volUpBtn.update(mPos);
            volDownBtn.update(mPos);
            backSettingsBtn.update(mPos);
            // Update Volume Text Display
            volumeDisplay.setString("Vol: " + to_string((int)musicVolume));
            FloatRect vb = volumeDisplay.getLocalBounds();
            // Position of the - and + buttons
            float btnY = volDownBtn.shape.getPosition().y;
            float btnH = volDownBtn.shape.getSize().y;
            volumeDisplay.setOrigin({ vb.size.x / 2.0f, vb.size.y / 2.0f });
            volumeDisplay.setPosition({ WINDOW_WIDTH / 2.0f, btnY + btnH / 2.0f
                });
        }
        else if (currentState == SELECT_DIFFICULTY) {
            easyBtn.update(mPos);
            mediumBtn.update(mPos);
            hardBtn.update(mPos);
        }
        else if (currentState == SET_LIMIT) {
            if (!isTypingCustomAmount) customLimitBtn.update(mPos);
            limitAllBtn.update(mPos);
            if (isTypingCustomAmount) confirmLimitBtn.update(mPos);
        }
        else if (currentState == PAUSED) {
            startBtn.update(mPos);
            endQuizBtn.update(mPos);
            exitBtn.update(mPos);
        }
        else if (currentState == GAME_OVER) {
            startBtn.update(mPos);
            exitBtn.update(mPos);
        }
        else if (currentState == QUIZ_MODE && !isAnswerLocked) {
            for (int i = 0; i < 4; ++i) {
                options[i].update(mPos);
            }
            skipBtn.update(mPos);
            backBtn.update(mPos);
            pauseBtn.update(mPos);
            quizCamBtn.update(mPos);
        }
        // Particles
        for (auto& p : particles) { // Updates position of particles using reference
            p.shape.move(p.velocity * dt); // Uses speed of particles to easily delete them by checking their timeframes
            p.lifetime -= dt; // Deletes 0.2s from the 1s lifetime of particles
            Color c = p.shape.getFillColor(); // Fills the color of Particles
            c.a = static_cast<uint8_t>(max(0.0f, p.lifetime * 255)); // With time, increase the transparency (255 = solid, 0 = transparent)
            p.shape.setFillColor(c);
        }
        erase_if(particles, [](const Particle& p) { return p.lifetime <= 0; }); // Deletes the particles

        // FloatingText
        for (auto& ft : floatTexts) {
            ft.text.move({ 0, -ft.speed * dt }); // Move Up
            ft.lifetime -= dt;
            // Fade out
            Color c = ft.text.getFillColor();
            c.a = static_cast<uint8_t>(max(0.0f, (ft.lifetime / 1.0f) * 255));
            ft.text.setFillColor(c);
            ft.text.setOutlineColor(Color(0, 0, 0, c.a)); // Fade outline
        }
        erase_if(floatTexts, [](const FloatingText& ft) { return ft.lifetime <= 0; }); // Deletes the text

        // Quiz Timer
        if (currentState == QUIZ_MODE) {
            if (!isAnswerLocked) {
                timeLeft -= dt;
                float ratio = timeLeft / TIME_PER_QUESTION;
                if (ratio < 0)
                    ratio = 0;
                float maxWidth = (float)WINDOW_WIDTH - 100.f;
                float currentWidth = maxWidth * ratio;
                timerBar.setSize({ currentWidth, 20.f });
                if (ratio > 0.5f) {
                    timerBar.setFillColor(Color::Green); // Green in the beginning
                }
                else if (ratio > 0.25f && ratio <= 0.5f) {
                    timerBar.setFillColor(Color(255, 165, 0)); // Orange when 7.5 seconds
                }
                else {
                    timerBar.setFillColor(Color::Red); // Red when 3.75 seconds
                    if ((int)(timeLeft * 10) % 2 == 0) timerBar.setFillColor(Color(200, 0, 0)); // Make it blink if very low
                    if (timeLeft <= 0) { // If time ends an duser didn't select an option
                        isAnswerLocked = true;
                        const auto& currentQ = allQuestions[currentQuestionIndex]; // Show the correct answer
                        options[currentQ.correctAnswerIndex].setColor(CORRECT_COLOR); // Change the correct answer to Green
                        shakeTime = 0.5f; // Shake
                        comboStreak = 0; // Resets Combo Streak
                        correctSound.setPitch(1.0f); // Resets Pitch
                        if (hasSound) incorrectSound.play(); // Play incorrect buzzer
                        feedbackTimer.restart(); // Timer Reset
                    }
                }
            }
            else if (autoNext && feedbackTimer.getElapsedTime() >= feedbackDuration) { // Next Question
                currentQuestionIndex++;
                loadQuestion();
            }
        }
        // Screen Shake
        if (shakeTime > 0) {
            shakeTime -= dt;
            float offsetX = (rand() % 200 - 100) / 100.0f * shakeMagnitude;
            float offsetY = (rand() % 200 - 100) / 100.0f * shakeMagnitude;
            shakeView.setCenter({ WINDOW_WIDTH / 2.0f + offsetX, WINDOW_HEIGHT / 2.0f + offsetY });
            window.setView(shakeView);
        }
        else {
            window.setView(originalView); // Resets to normal
        }

        // Fade Logic
        if (fadeAlpha > 0) {
            fadeAlpha -= 500.0f * dt; // Fade speed
            if (fadeAlpha < 0) fadeAlpha = 0;
        }

        /*-------------------------  Drawing  -------------------------*/

        window.clear(BACKGROUND_COLOR);

        // Background
        if (backgroundSprite) {
            backgroundSprite->setColor(animatedBgColor);
            window.draw(*backgroundSprite);
        }

        // Particles
        for (const auto& p : particles) {
            window.draw(p.shape);
        }

        // Floating Text
        for (const auto& ft : floatTexts) {
            window.draw(ft.text);
        }

        // UI States
        if (currentState == MENU) {
            titleText.setString("C++ Logic Builder");
            titleText.setCharacterSize(55);
            FloatRect tRect = titleText.getLocalBounds();
            titleText.setOrigin({ tRect.position.x + tRect.size.x / 2.0f, tRect.position.y + tRect.size.y / 2.0f });
            titleText.setPosition({ WINDOW_WIDTH / 2.0f, 160.0f });
            Text shadow = titleText;
            shadow.setFillColor(Color(0, 0, 0, 150)); // Black with transparency Shadow effect
            shadow.move({ 4.0f, 4.0f }); // Shift shadow down-right
            window.draw(shadow);
            window.draw(titleText);

            Text highScoreText(uifont);
            highScoreText.setString("High Score: " + to_string(getHighScore()));
            highScoreText.setCharacterSize(30);
            highScoreText.setFillColor(Color::Yellow);
            FloatRect hsRect = highScoreText.getLocalBounds(); // Centering the Text
            highScoreText.setOrigin({ hsRect.size.x / 2.0f, 0 }); // Origin Setting of the Text
            highScoreText.setPosition({ WINDOW_WIDTH / 2.0f, 50.0f });
            shadow = highScoreText;
            shadow.setFillColor(Color(0, 0, 0, 150));
            shadow.move({ 4.0f, 4.0f });
            window.draw(shadow);
            window.draw(highScoreText);

            Text sub(uifont, "MASTER THE SKILL!", 24);
            FloatRect sb = sub.getLocalBounds();
            sub.setOrigin({ sb.size.x / 2, 0 });
            sub.setPosition({ WINDOW_WIDTH / 2.0f, 220.f });
            window.draw(sub);

            startBtn.setOptionText("Start Game");
            startBtn.setPosition({ WINDOW_WIDTH / 2.0f - 120.0f, 300.0f });
            startBtn.draw(window);
            settingsBtn.setOptionText("Settings");
            settingsBtn.setPosition({ WINDOW_WIDTH / 2.0f - 120.0f, 380.0f });
            settingsBtn.draw(window);
            exitBtn.setOptionText("Exit Game");
            exitBtn.setPosition({ WINDOW_WIDTH / 2.0f - 120.0f, 460.0f });
            exitBtn.draw(window);
            Text credits(uifont);
            credits.setString("Created by Muhammad Faizan | End Semester Project");
            credits.setCharacterSize(18);
            credits.setFillColor(Color(255, 255, 255, 255)); // White Text
            FloatRect crRect = credits.getLocalBounds();
            credits.setOrigin({ crRect.size.x, crRect.size.y }); // Align to bottom-right
            credits.setPosition({ WINDOW_WIDTH - 20.0f, WINDOW_HEIGHT - 20.0f });
            window.draw(credits);
        }
        else if (currentState == SETTINGS) {
            titleText.setString("Audio Settings");
            FloatRect tRect = titleText.getLocalBounds();
            titleText.setOrigin({ tRect.position.x + tRect.size.x / 2.0f, tRect.position.y + tRect.size.y / 2.0f });
            titleText.setPosition({ WINDOW_WIDTH / 2.0f, 160.0f });
            Text shadow = titleText;
            shadow.setFillColor(Color(0, 0, 0, 150));
            shadow.move({ 4.0f, 4.0f });
            window.draw(shadow);
            window.draw(titleText);
            // Drawing all Buttons in Settings
            toggleMusicBtn.draw(window);
            toggleSfxBtn.draw(window);
            toggleCamBtn.draw(window);
            volDownBtn.draw(window);
            window.draw(volumeDisplay);
            volUpBtn.draw(window);
            backSettingsBtn.draw(window);
        }
        else if (currentState == SELECT_DIFFICULTY) {
            titleText.setString("Select Difficulty");
            FloatRect tRect = titleText.getLocalBounds();
            titleText.setOrigin({ tRect.position.x + tRect.size.x / 2.0f, tRect.position.y + tRect.size.y / 2.0f });
            titleText.setPosition({ WINDOW_WIDTH / 2.0f, 160.0f });
            Text shadow = titleText;
            shadow.setFillColor(Color(0, 0, 0, 150)); // Black shadow
            shadow.move({ 4.0f, 4.0f });
            window.draw(shadow);
            window.draw(titleText);
            easyBtn.draw(window);
            mediumBtn.draw(window);
            hardBtn.draw(window);
        }
        else if (currentState == SET_LIMIT) {
            titleText.setString(currentDifficultyName);
            titleText.setPosition({ WINDOW_WIDTH / 2.0f, 150.0f });
            titleText.setCharacterSize(55); // Make it slightly larger
            titleText.setFillColor(Color(180, 200, 255)); // Light Blue tint

            // Perfect Centering
            FloatRect tr = titleText.getLocalBounds();
            titleText.setOrigin({ tr.position.x + tr.size.x / 2.0f, tr.position.y + tr.size.y / 2.0f });
            titleText.setPosition({ WINDOW_WIDTH / 2.0f, 150.0f });

            // Shadow Effect
            Text titleShadow = titleText;
            titleShadow.setFillColor(Color(0, 0, 0, 180)); // Dark semi-transparent black
            titleShadow.move({ 5.0f, 5.0f }); // Shift 5 pixels down and right

            window.draw(titleShadow); // Draw shadow Behind the text
            window.draw(titleText);

            Text prompt(uifont);
            prompt.setString("Questions Available: " + to_string(totalQuestions));
            prompt.setCharacterSize(24);
            prompt.setFillColor(Color::White);
            FloatRect pr = prompt.getLocalBounds();
            prompt.setOrigin({ pr.size.x / 2, 0 });
            prompt.setPosition({ WINDOW_WIDTH / 2.0f, 230.0f });
            window.draw(prompt);

            if (isTypingCustomAmount) {
                customLimitBtn.shape.setFillColor(Color(40, 40, 40));
                customLimitBtn.shape.setOutlineColor(Color(180, 200, 255));
                window.draw(customLimitBtn.shape);
                bool showCursor = (int)(effectClock.getElapsedTime().asSeconds() * 2.0f) % 2 == 0;
                if (showCursor) {
                    customInputDisplay.setString(customInputString + "|");
                }
                else {
                    customInputDisplay.setString(customInputString);
                }
                customInputDisplay.setFillColor(Color(180, 200, 255));
                customInputDisplay.setCharacterSize(30);
                FloatRect bounds = customInputDisplay.getLocalBounds();
                Vector2f btnCenter = customLimitBtn.shape.getPosition() + (customLimitBtn.shape.getSize() / 2.0f);
                customInputDisplay.setOrigin({ bounds.position.x + bounds.size.x / 2.0f, bounds.position.y + bounds.size.y / 2.0f });
                customInputDisplay.setPosition(btnCenter);
                window.draw(customInputDisplay);

                Text sub(uifont, "Type amount & Press ENTER", 18);
                sub.setFillColor(Color::Yellow);
                FloatRect subRect = sub.getLocalBounds();
                sub.setOrigin({ subRect.position.x + subRect.size.x / 2.0f, 0.f });
                sub.setPosition({ btnCenter.x, btnCenter.y + 45 });
                window.draw(sub);
                confirmLimitBtn.draw(window);
            }
            else {
                customLimitBtn.setColor(UI_BASE_COLOR);
                customLimitBtn.draw(window);
            }
            limitAllBtn.draw(window);
        }
        else if (currentState == QUIZ_MODE || currentState == PAUSED) {
            titleText.setString(currentDifficultyName);
            titleText.setPosition({ WINDOW_WIDTH / 2.0f, 60.0f });
            FloatRect b = titleText.getLocalBounds();
            titleText.setOrigin({ b.position.x + b.size.x / 2.0f, 0.0f });
            // Draw Shadow
            Text tShadow = titleText;
            tShadow.setFillColor(Color(0, 0, 0, 150)); // Transparent Black
            tShadow.move({ 3.0f, 3.0f });              // Shift 3 pixels down-right
            window.draw(tShadow);
            window.draw(titleText);
            window.draw(timerTrack);
            window.draw(timerBar);
            window.draw(questionText);
            for (int i = 0; i < 4; ++i) options[i].draw(window);
            scoreText.setString("Question: " + to_string(currentQuestionIndex + 1) + "/" + to_string(actualTotalQuestions) + " | Score: " + to_string(score));
            window.draw(scoreText);
            skipBtn.draw(window);
            backBtn.draw(window);
            pauseBtn.draw(window);
            quizCamBtn.draw(window);

            if (currentState == PAUSED) {
                RectangleShape overlay({ WINDOW_WIDTH, WINDOW_HEIGHT });
                overlay.setFillColor(Color(0, 0, 0, 200));
                window.draw(overlay);
                Text pauseTitle(uifont, "PAUSED", 60);
                pauseTitle.setFillColor(Color::White);
                FloatRect ptRect = pauseTitle.getLocalBounds();
                pauseTitle.setOrigin({ ptRect.position.x + ptRect.size.x / 2.0f, 0 });
                pauseTitle.setPosition({ WINDOW_WIDTH / 2.0f, 150.0f });
                window.draw(pauseTitle);

                startBtn.setOptionText("Resume Game (Esc)");
                startBtn.setPosition({ WINDOW_WIDTH / 2.0f - 120.0f, 300.0f });
                startBtn.draw(window);
                endQuizBtn.setPosition({ WINDOW_WIDTH / 2.0f - 120.0f, 380.0f });
                endQuizBtn.draw(window);
                exitBtn.setOptionText("Exit to Main Menu");
                exitBtn.setPosition({ WINDOW_WIDTH / 2.0f - 120.0f, 460.0f });
                exitBtn.draw(window);
            }
        }
        else if (currentState == GAME_OVER) {
            saveHighScore(score);
            titleText.setString("QUIZ COMPLETE!");
            titleText.setCharacterSize(60);
            FloatRect tRect = titleText.getLocalBounds();
            titleText.setOrigin({ tRect.position.x + tRect.size.x / 2.0f, tRect.position.y + tRect.size.y / 2.0f });
            titleText.setPosition({ WINDOW_WIDTH / 2.0f, 150.0f });
            Text tShadow = titleText;
            tShadow.setFillColor(Color(0, 0, 0, 150));
            tShadow.move({ 4.0f, 4.0f });
            window.draw(tShadow);
            window.draw(titleText);

            string finalScoreString = "Final score: " + to_string(score) + " / " + to_string(actualTotalQuestions);
            Text finalScore(uifont, finalScoreString, 40);
            FloatRect fsRect = finalScore.getLocalBounds();
            finalScore.setOrigin({ fsRect.position.x + fsRect.size.x / 2.0f, 0 });
            finalScore.setPosition({ WINDOW_WIDTH / 2.0f, 230.0f });
            window.draw(finalScore);

            startBtn.setOptionText("Back to Menu");
            startBtn.setPosition({ WINDOW_WIDTH / 2.0f - 120.0f, 300.0f });
            startBtn.draw(window);
            exitBtn.setOptionText("Exit Game");
            exitBtn.setPosition({ WINDOW_WIDTH / 2.0f - 120.0f, 380.0f });
            exitBtn.draw(window);
        }
        // Draw Fade Overlay
        if (fadeAlpha > 0) {
            fadeRect.setFillColor(Color(0, 0, 0, static_cast<uint8_t>(fadeAlpha)));
            window.draw(fadeRect);
        }
        window.display();
    }
    return 0;
}

OptionButton::OptionButton(float x, float y, float w, float h, const string& prefixText, const Font& font)
    : text(font), prefix(font)
{
    originalPos = { x, y };
    shape.setPosition({ x, y });
    shape.setSize({ w, h });
    shape.setCornersRadius(15.0f);
    shape.setFillColor(baseFillColor);
    shape.setOutlineThickness(0);
    prefix.setString(prefixText);
    prefix.setCharacterSize(28);
    prefix.setFillColor(Color::Yellow);
    float textVerticalOffset = (h / 2.0f) - (prefix.getCharacterSize() / 2.0f) - 5;
    prefix.setPosition({ x + 15, y + textVerticalOffset });
    text.setCharacterSize(20);
    text.setFillColor(Color::White);
    if (prefixText != "A:" && prefixText != "B:" && prefixText != "C:" && prefixText != "D:") { setOptionText(prefixText); }
}
void OptionButton::update(Vector2i mousePos) {
    if (isClicked(mousePos)) {
        // Brighten Background
        shape.setFillColor(Color(
            min(baseFillColor.r + 50, 255),
            min(baseFillColor.g + 50, 255),
            min(baseFillColor.b + 50, 255),
            255
        ));
        // Change Font Color
        text.setFillColor(Color::White);
        prefix.setFillColor(Color::Yellow);

        // Add subtle glow (Shadow)
        shape.setOutlineThickness(2);
        shape.setOutlineColor(Color(255, 255, 255, 100));
    }
    else {
        // Return to ground
        setPosition(originalPos);
        // Reset Colors
        shape.setFillColor(baseFillColor);
        text.setFillColor(Color::White);
        prefix.setFillColor(Color::Yellow);
        shape.setOutlineThickness(0);
    }
}
void OptionButton::setOptionText(const string& optionText) {
    text.setString(optionText); FloatRect textBounds = text.getLocalBounds();
    float buttonWidth = shape.getSize().x; float buttonHeight = shape.getSize().y; float shapeX = shape.getPosition().x; float shapeY = shape.getPosition().y;
    float newX = prefix.getString().isEmpty() ? shapeX + (buttonWidth / 2.0f) - (textBounds.size.x / 2.0f) : shapeX + 70;
    float newY = shapeY + (buttonHeight / 2.0f) - (text.getCharacterSize() / 2.0f) - 5;
    text.setPosition({ newX, newY });
}
void OptionButton::draw(RenderWindow& window) const {
    window.draw(shape);
    Text tShadow = text; tShadow.setFillColor(Color(0, 0, 0, 150)); tShadow.move({ 3.0f, 3.0f }); window.draw(tShadow);
    if (!prefix.getString().isEmpty()) { Text pShadow = prefix; pShadow.setFillColor(Color(0, 0, 0, 150)); pShadow.move({ 3.0f, 3.0f }); window.draw(pShadow); }
    window.draw(prefix); window.draw(text);
}
void OptionButton::setPosition(const Vector2f& pos) {
    shape.setPosition(pos);
    originalPos = pos;
    prefix.setPosition({ pos.x + 15, pos.y + (shape.getSize().y / 2.0f) - (prefix.getCharacterSize() / 2.0f) - 5 });
    setOptionText(text.getString());
}
// Load Function
vector<QuizQuestion> loadQuestionsFromFile(const string& filename) {
    vector<QuizQuestion> questions;
    ifstream file(filename);
    if (!file.is_open()) return questions; // Return empty if failed

    string line;
    while (getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        stringstream ss(line);
        string segment;
        vector<string> parts;
        while (getline(ss, segment, '|')) { parts.push_back(segment); }
        if (parts.size() == 6) {
            try {
                QuizQuestion q;
                q.questionText = parts[0];
                q.options = { parts[1], parts[2], parts[3], parts[4] };
                size_t pos = 0;
                while ((pos = q.questionText.find("\\n", pos)) != string::npos) {
                    q.questionText.replace(pos, 2, "\n");
                    pos += 1;
                }
                q.correctAnswerIndex = stoi(parts[5]);
                if (q.correctAnswerIndex >= 0 && q.correctAnswerIndex < 4) questions.push_back(q);
            }
            catch (...) {}
        }
    }
    return questions;
}
void spawnParticles(vector<Particle>& particles, Vector2f pos, Color color) {
    for (int i = 0; i < 20; i++) { // Spawn 20 particles
        Particle p;
        p.shape.setSize({ 8.f, 8.f });
        p.shape.setPosition(pos);
        p.shape.setFillColor(color);
        p.lifetime = 1.0f; // Lasts 1 second

        // Random velocity
        float angle = (rand() % 360) * 3.14159f / 180.f;
        float speed = (rand() % 150 + 50); // Speed between 50 and 200
        p.velocity = { cos(angle) * speed, sin(angle) * speed };

        particles.push_back(p);
    }
}
int getHighScore() {
    ifstream input("highscore.txt");
    int highScore = 0;
    if (input.is_open()) {
        input >> highScore;
    }
    return highScore;
}
void saveHighScore(int currentScore) {
    int oldHigh = getHighScore();
    if (currentScore > oldHigh) {
        ofstream output("highscore.txt");
        output << currentScore;
    }
}
