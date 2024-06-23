import 'package:cancer/auth/login/login_screen.dart';
import 'package:cancer/auth/register/register_screen.dart';
import 'package:cancer/my_theme.dart';
import 'package:cancer/screens/begin_screen.dart';
import 'package:cancer/screens/checkup_screen.dart';
import 'package:cancer/screens/deals.dart';
import 'package:cancer/screens/diagnosis.dart';
import 'package:cancer/screens/hend.dart';
import 'package:cancer/screens/hoda.dart';
import 'package:cancer/screens/know_more.dart';
import 'package:cancer/screens/managing_stress.dart';
import 'package:cancer/screens/risk_factors_screen.dart';
import 'package:cancer/screens/self_examination.dart';
import 'package:cancer/screens/settings.dart';
import 'package:cancer/screens/simona.dart';
import 'package:cancer/screens/success_screen.dart';
import 'package:cancer/screens/talking_children.dart';
import 'package:cancer/screens/talking_partner.dart';
import 'package:cancer/screens/talking_relatives.dart';
import 'package:cancer/taps/pages_screen.dart';
import 'package:firebase_core/firebase_core.dart';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:device_preview/device_preview.dart'; 
import 'package:flutter/foundation.dart'; 
import 'home/auth_provider.dart';
import 'home/home_screen.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await Firebase.initializeApp().catchError((error) {
    print("Firebase initialization error: $error");
  });

  runApp(DevicePreview(
    enabled: !kReleaseMode, 
    builder: (context) => MyApp(),
  ));
}

class MyApp extends StatelessWidget {
  const MyApp({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MultiProvider(
      providers: [
        ChangeNotifierProvider(create: (_) => UserProvider()),
      ],
      child: Builder(builder: (context) {
        return MaterialApp(
          debugShowCheckedModeBanner: false,
          locale: DevicePreview.locale(context), 
          builder: DevicePreview.appBuilder, 
          initialRoute: BeginScreen.routeName,
          routes: {
            LoginScreen.routeName: (context) => LoginScreen(),
            HomeScreen.routeName: (context) => HomeScreen(),
            RegisterScreen.routeName: (context) => RegisterScreen(),
            CheckUpScreen.routeName: (context) => CheckUpScreen(),
            SuccessScreen.routeName: (context) => SuccessScreen(),
            Deals.routeName: (context) => Deals(),
            DiagnosisScreen.routeName: (context) => DiagnosisScreen(),
            PagesScreen.routeName: (context) => PagesScreen(),
            KnowMore.routeName: (context) => KnowMore(),
            Settings.routeName: (context) => Settings(),
            RiskFactors.routeName: (context) => RiskFactors(),
            BeginScreen.routeName: (context) => BeginScreen(),
            Hind.routeName: (context) => Hind(),
            Hoda.routeName: (context) => Hoda(),
            Simona.routeName: (context) => Simona(),
            TalkingPartner.routeName: (context) => TalkingPartner(),
            TalkingChildren.routeName: (context) => TalkingChildren(),
            TalkingRelatives.routeName: (context) => TalkingRelatives(),
            ManagingStress.routeName: (context) => ManagingStress(),
            SelfExamination.routeName: (context) => SelfExamination(),
          },
          theme: MyTheme.lightTheme,
        );
      }),
    );
  }
}
